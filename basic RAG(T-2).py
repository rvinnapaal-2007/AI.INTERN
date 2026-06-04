import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.agents import create_agent

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# PDF Processing Function
# --------------------------------------------------
def prepare_documents(pdf_file):
    print("Loading PDF...")

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} document chunks.")

    return chunks


# --------------------------------------------------
# Create Vector Database
# --------------------------------------------------
def create_vector_database(chunks):
    print("Generating Embeddings...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_db


# --------------------------------------------------
# Load and Index PDF
# --------------------------------------------------
pdf_chunks = prepare_documents("R23_IT_Syllabus.pdf")
vector_store = create_vector_database(pdf_chunks)


# --------------------------------------------------
# Retrieval Tool
# --------------------------------------------------
@tool
def syllabus_search(question: str) -> str:
    """
    Search the syllabus PDF and return relevant content.
    """

    results = vector_store.similarity_search(
        question,
        k=4
    )

    output = []

    for doc in results:
        page_number = doc.metadata.get("page", "Unknown")

        output.append(
            f"""
Page Number: {page_number + 1}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(output)


# --------------------------------------------------
# LLM Configuration
# --------------------------------------------------
llm = ChatGroq(
    model="qwen/qwen3-32b",
    reasoning_format="parsed"
)


# --------------------------------------------------
# Agent Creation
# --------------------------------------------------
system_message = """
You are a university syllabus assistant.

Always use the syllabus_search tool whenever
the user's question is related to the PDF.

Provide concise and accurate answers.
"""

agent = create_agent(
    model=llm,
    tools=[syllabus_search],
    system_prompt=system_message
)


# --------------------------------------------------
# Interactive Chat Loop
# --------------------------------------------------
print("\n=== PDF Syllabus Assistant Ready ===")

while True:

    user_query = input("\nAsk a Question (type 'exit' to quit): ")

    if user_query.lower() == "exit":
        print("Session Ended.")
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        }
    )

    print("\nAnswer:\n")

    print(response["messages"][-1].content)
