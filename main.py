
import os
import shutil
import asyncpg
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Modern imports for creating agents
from langchain.agents import AgentExecutor, create_react_agent

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is missing from your .env profile.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# CRITICAL DEBUG CHECK FOR 401 ERRORS:
print(f"--- [STARTUP DEBUG] GROQ_API_KEY length detected in system: {len(GROQ_API_KEY) if GROQ_API_KEY else 0} characters ---")

# Updated model to the powerful 'llama-3.3-70b-versatile' model on Groq
MODEL = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.6
)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_pool = None
USER_VECTOR_STORES = {}

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def rebuild_vector_stores():
    """On startup, re-index any PDFs saved to disk so memory survives restarts."""
    if not os.path.exists(UPLOAD_DIR):
        return
    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith(".pdf"):
            user_id = filename.split("_", 1)[0]
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(docs)
                USER_VECTOR_STORES[user_id] = InMemoryVectorStore.from_documents(splits, EMBEDDINGS)
                print(f" Re-indexed PDF for user: {user_id}")
            except Exception as e:
                print(f" Failed to re-index {filename}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    try:
        # 1. Establish PostgreSQL connection pool
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=15)
        print(" PostgreSQL connection pool established.")
        
        # 2. Automatically create tables if they do not exist
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                print(" Verifying database schema...")
                
                # Create chat sessions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create chat messages table linked to sessions via foreign key
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        sender VARCHAR(50) NOT NULL CHECK (sender IN ('user', 'ai')),
                        message_text TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                print(" Database tables verified/created successfully.")

        # 3. Re-index existing local PDF structures
        await rebuild_vector_stores()
        
    except Exception as e:
        print(f" Startup failure: {str(e)}")
        raise e
    yield
    if db_pool:
        await db_pool.close()
        print(" PostgreSQL connection pool closed.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, f"{user_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        extracted_text = "".join(
            doc.page_content.strip()
            for doc in docs
        )

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in PDF."
            )

        if len(extracted_text) < 50:
            raise HTTPException(
                status_code=400,
                detail="PDF contains insufficient text."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = splitter.split_documents(docs)
        USER_VECTOR_STORES[user_id] = InMemoryVectorStore.from_documents(splits, EMBEDDINGS)
        return {"status": "success", "message": f"Indexed: {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{user_id}")
async def get_user_sessions(user_id: str):
    """Returns all sessions and their messages for a given user on page load."""
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    async with db_pool.acquire() as conn:
        sessions = await conn.fetch(
            "SELECT session_id FROM chat_sessions WHERE user_id = $1 ORDER BY created_at ASC",
            user_id
        )
        result = {}
        for s in sessions:
            sid = str(s["session_id"])
            messages = await conn.fetch(
                """
                SELECT sender, message_text 
                FROM chat_messages 
                WHERE session_id = $1 
                ORDER BY created_at ASC
                """,
                sid
            )
            result[sid] = [{"sender": r["sender"], "text": r["message_text"]} for r in messages]
        return {"sessions": result, "has_pdf": user_id in USER_VECTOR_STORES}


@app.post("/chat")
async def chat(
    message: str = Form(...),
    user_id: str = Form(...),
    session_id: str = Form(...)
):
    if user_id not in USER_VECTOR_STORES:
        raise HTTPException(status_code=400, detail="No document found. Please upload a PDF first.")

    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")

    # --- STEP 1: SAVE USER MESSAGE & FETCH HISTORY (QUICK DB HIT) ---
    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():  # FORCE COMMIT
                await conn.execute(
                    "INSERT INTO chat_sessions (session_id, user_id) VALUES ($1, $2) ON CONFLICT (session_id) DO NOTHING",
                    session_id, user_id
                )
                await conn.execute(
                    "INSERT INTO chat_messages (session_id, sender, message_text) VALUES ($1, $2, $3)",
                    session_id, "user", message
                )

                rows = await conn.fetch(
                    "SELECT sender, message_text FROM chat_messages WHERE session_id = $1 ORDER BY created_at ASC",
                    session_id
                )
    except Exception as e:
        print(f"Database error saving user message: {e}")
        raise HTTPException(status_code=500, detail="Failed to log message to history.")

    # --- STEP 2: PREPARE AGENT & CALL LLM (NO DATABASE CONNECTION POOL HELD) ---
    formatted_history = []
    for r in rows:
        if r["sender"] == "user":
            formatted_history.append(HumanMessage(content=r["message_text"]))
        else:
            formatted_history.append(AIMessage(content=r["message_text"]))

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieves relevant context from the uploaded PDF document."""
        store = USER_VECTOR_STORES[user_id]
        results = store.similarity_search_with_score(query, k=3)
        if not results:
            return "NO_RELEVANT_CONTEXT"

        docs = [doc for doc, score in results]
        return "\n\n".join(
            f"Content: {d.page_content}\nSource: {d.metadata.get('source', 'unknown')}"
            for d in docs
        )

    tools = [retrieve_context]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document assistant.
You must answer ONLY using information retrieved from the uploaded PDF using your tools.
If the tool returns NO_RELEVANT_CONTEXT, respond exactly:
I could not find relevant information about this in the uploaded document.

You have access to the following tools:
{tools}

To use a tool, you MUST use the exact format below. Do not add any extra conversational text before this format:

Thought: Do I need to use a tool? Yes
Action: {tool_names}
Action Input: the input to the action
Observation: the result of the action

When you have the final answer to give to the user, you MUST use this exact format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}\n\n{agent_scratchpad}")
    ])

    agent = create_react_agent(MODEL, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    try:
        response = agent_executor.invoke({
            "input": message,
            "chat_history": formatted_history[:-1]  # Strip out current user message for historical consistency
        })
        ai_message = response["output"]
    except Exception as e:
        print(f"Agent Execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # --- STEP 3: SAVE AI RESPONSE (QUICK DB HIT) ---
    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():  # FORCE COMMIT
                await conn.execute(
                    "INSERT INTO chat_messages (session_id, sender, message_text) VALUES ($1, $2, $3)",
                    session_id, "ai", ai_message
                )
    except Exception as e:
        print(f"Database error saving AI response: {e}")
    
    return {"response": ai_message}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool unavailable.")
    
    async with db_pool.acquire() as conn:
        try:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM chat_messages WHERE session_id = $1",
                    session_id
                )
                await conn.execute(
                    "DELETE FROM chat_sessions WHERE session_id = $1",
                    session_id
                )
            return {"status": "success", "message": "Session deleted successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))