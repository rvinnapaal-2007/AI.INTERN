import os
from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b", reasoning_format="parsed")
print("Loading PDF Doc...")
loader = PyPDFLoader('R23_IT_Syllabus.pdf')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
all_splits  = text_splitter.split_documents(docs)
 
print("Generating embeddings....")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = InMemoryVectorStore.from_documents(all_splits,embeddings)

@tool
def retrieve_context(query : str)->str:
    """Retrieves relevant context from the PDf document based on the query."""
    similar_docs = vector_store.similarity_search(query,k=5 )
    data = []
    for doc in similar_docs:
        content = doc.page_content
        source = doc.metadata.get("source","unknown")
        data.append(f"Content: {content}\nSource: {source}")
    return "\n".join(data)

tools = [retrieve_context]
prompt = "You are an agent who retrieves data from PDF."
agent = create_agent(model, [retrieve_context],system_prompt=prompt)
query = "who is india PM ?"

for step in agent.stream({"messages": [
    {
        'role':'user',
        'content':query
    }
]}, stream_mode="values"):
    step["messages"][-1].pretty_print()
