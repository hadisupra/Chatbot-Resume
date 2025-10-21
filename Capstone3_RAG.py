# importing all libraries needed
from itertools import chain
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import pickle
import langgraph
# import sklearn
from qdrant_client import QdrantClient
import streamlit as st
from langchain_community.docstore.in_memory import InMemoryDocstore #for allocating vector database storage (in RAM)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI #for embedding and LLM service
from langchain_core.tools import tool #import tool decorator
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings #import huggingface embedding
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings import OpenAIEmbeddings #import openai embedding
from langchain.chains import RetrievalQA #import retrievalQA chain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage #import message type
from pydantic import BaseModel, Field #pydantic BaseModel and Field to create a data validation and data type definition
from typing import List, Any #for list type and dynamic typing
from langgraph.prebuilt import create_react_agent #import prebuilt-react agent from langgraph
from datetime import datetime #for date and time
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate #import prompt template
import sqlalchemy #import sql alchemy for database connection
from sqlalchemy import create_engine #import create_engine for database connection
from dotenv import load_dotenv #import load_dotenv to load environment variable from .env file
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
"""
Qdrant-backed Resume Assistant (RAG + ReAct Agent)
"""

import os
import pandas as pd
import streamlit as st
from typing import List

from dotenv import load_dotenv
from datetime import datetime

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
import streamlit as st



# ---- UI Header ----#
st.set_page_config(page_title="Resume Assistant", page_icon="🧠", layout="centered")
load_dotenv()


# ---- Credentials ---- (prefer st.secrets -> env -> input)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))

# Optional configuration from secrets/env with sensible defaults
QDRANT_COLLECTION = st.secrets.get("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "resumes"))
MODEL_CHAT = st.secrets.get("MODEL_CHAT", os.getenv("MODEL_CHAT", "gpt-4o-mini"))
MODEL_EMBEDDING = st.secrets.get("MODEL_EMBEDDING", os.getenv("MODEL_EMBEDDING", "text-embedding-3-small"))
CHUNK_SIZE = int(st.secrets.get("CHUNK_SIZE", os.getenv("CHUNK_SIZE", 1000)))
CHUNK_OVERLAP = int(st.secrets.get("CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP", 200)))
TOP_K = int(st.secrets.get("TOP_K", os.getenv("TOP_K", 5)))
RESUME_CSV_PATH = st.secrets.get("RESUME_CSV_PATH", os.getenv("RESUME_CSV_PATH", "Resume.csv"))

col1, col2 = st.columns(2)
with col1:
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
        if OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY
with col2:
    if not QDRANT_URL:
        QDRANT_URL = st.text_input("Qdrant URL (https://...)")
        if QDRANT_URL:
            os.environ["QDRANT_URL"] = QDRANT_URL
            st.session_state["QDRANT_URL"] = QDRANT_URL

if not QDRANT_API_KEY:
    QDRANT_API_KEY = st.text_input("Qdrant API Key", type="password")
    if QDRANT_API_KEY:
        os.environ["QDRANT_API_KEY"] = QDRANT_API_KEY
        st.session_state["QDRANT_API_KEY"] = QDRANT_API_KEY

st.divider()

# Reuse existing collection option
reuse_existing = st.checkbox("Reuse existing Qdrant collection (skip re-indexing)", value=True)


# ---- Load Dataset ----
@st.cache_data(show_spinner=False)
def load_resume_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    # Prefer 'Resume' column if present; else build from available columns
    if "Resume" in df.columns:
        df["text"] = df["Resume"].astype(str)
    else:
        cols = [c for c in ["Resume_str", "Category"] if c in df.columns]
        if not cols:
            raise ValueError("Dataset must contain a 'Resume' column or ['Resume_str','Category']")
        df["text"] = df[cols].fillna("").agg(" ".join, axis=1)
    return df[["text"]]

try:
    df = load_resume_df(RESUME_CSV_PATH)
except Exception as e:
    st.error(f"Error loading Resume.csv: {e}")
    st.stop()

# ---- Build Qdrant Vector Store ----
def build_qdrant_store(texts: List[str], qdrant_url: str, qdrant_key: str):
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text("\n".join(texts))
    embeddings = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    store = Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_key,
        collection_name=QDRANT_COLLECTION,
    )
    return store



if not (OPENAI_API_KEY and QDRANT_URL and QDRANT_API_KEY):
    st.warning("Please provide OpenAI API Key, Qdrant URL, and Qdrant API Key to proceed.")
    st.stop()

# Diagnostics expander
with st.expander("Diagnostics (optional)"):
    if st.button("Run diagnostics"):
        # Test OpenAI embeddings
        try:
            _ = OpenAIEmbeddings(model="text-embedding-3-small").embed_query("hello world")
            st.success("OpenAI Embeddings OK")
        except Exception as e:
            st.error(f"OpenAI Embeddings error: {e}")

        # Test LLM
        try:
            _ = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke("ping")
            st.success("OpenAI Chat LLM OK")
        except Exception as e:
            st.error(f"OpenAI Chat error: {e}")

        # Test Qdrant connectivity
        try:
            qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            cols = qc.get_collections()
            st.success(f"Qdrant OK. Collections: {[c.name for c in getattr(cols, 'collections', [])]}")
        except Exception as e:
            st.error(f"Qdrant connectivity error: {e}")

# Build or reuse the vector store with clear error surfacing
vector_store = None
if reuse_existing:
    try:
        # Check if collection exists
        qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        cols = qc.get_collections()
        existing = any(getattr(c, "name", "") == QDRANT_COLLECTION for c in getattr(cols, "collections", []))
    except Exception as e:
        existing = False
        st.info(f"Could not verify collections (will try to index). Details: {e}")

    if existing:
        try:
            with st.spinner(f"Connecting to existing collection '{QDRANT_COLLECTION}'..."):
                vector_store = Qdrant.from_existing_collection(
                    embedding=OpenAIEmbeddings(model=MODEL_EMBEDDING),
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    collection_name=QDRANT_COLLECTION,
                )
            st.success(f"Using existing collection: {QDRANT_COLLECTION}")
        except Exception as e:
            st.warning(f"Failed to use existing collection; will re-index. Details: {e}")

if vector_store is None:
    with st.spinner("Indexing resumes in Qdrant (first time may take a minute)..."):
        try:
            vector_store = build_qdrant_store(df["text"].astype(str).tolist(), QDRANT_URL, QDRANT_API_KEY)
            st.success(f"Indexed into collection: {QDRANT_COLLECTION}")
        except Exception as e:
            st.error(f"Failed to build Qdrant vector store: {e}")
            st.stop()
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

# ---- QA Chain ----
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=MODEL_CHAT, temperature=0),
    chain_type="stuff",
    retriever=retriever,
)

# ---- Tools ----
@tool
def current_datetime(_=None) -> str:
    """Return the current date and time (YYYY-MM-DD HH:MM:SS)."""
    from datetime import datetime as _dt
    return _dt.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_resumes(query: str) -> str:
    """Retrieve top resume chunks relevant to the query."""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in docs[:5]])

@tool
def answer_from_resumes(question: str) -> str:
    """Answer a question using the resume knowledge base via RAG."""
    return qa_chain.run(question)

# ---- Agent ----
agent = create_react_agent(
    model=ChatOpenAI(model=MODEL_CHAT, temperature=0),
    tools=[current_datetime, search_resumes, answer_from_resumes],
    prompt=SystemMessage(
        content=(
            "You are a helpful Resume Assistant. Use tools to search and answer from resumes. "
            "Prefer 'answer_from_resumes' for final answers. Be concise and cite snippets when helpful."
        )
    ),
)

# st.set_page_config(
#     page_title="Resume Assistant", 
#     page_icon="🧠"
# )
# st.title("Resume Assistant (Qdrant RAG + Agent)")
# st.markdown("Ask questions about your resume dataset. The assistant retrieves relevant chunks and reasons with a ReAct agent.")

# ---- Chat UI ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

if user_q := st.chat_input("Ask about the resumes (skills, roles, years, etc.)"):
    with st.chat_message("Human"):
        st.markdown(user_q)
    st.session_state.messages.append({"role": "Human", "content": user_q})

    with st.chat_message("AI"):
        response = agent.invoke({"messages": [HumanMessage(content=user_q)]})
        answer = response["messages"][-1].content
        st.markdown(answer)
    st.session_state.messages.append({"role": "AI", "content": answer})
