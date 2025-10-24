# importing all libraries needed
import os
import json
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
# Version-robust imports: prefer new APIs, fallback later
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
except Exception:  # pragma: no cover - older langchain
    create_stuff_documents_chain = None  # type: ignore
try:
    from langchain.chains import create_retrieval_chain  # type: ignore
except Exception:  # pragma: no cover - older langchain
    create_retrieval_chain = None  # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain_core.output_parsers import StrOutputParser  # type: ignore
except Exception:  # pragma: no cover
    StrOutputParser = None  # type: ignore
try:
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # type: ignore
except Exception:  # pragma: no cover
    RunnableLambda = None  # type: ignore
    RunnablePassthrough = None  # type: ignore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from langchain.agents import create_agent


 # ---- UI Header ----#
st.set_page_config(page_title="Resume Assistant", page_icon="🧠", layout="centered")
st.title("💬 Resume Advisor Chatbot")
st.markdown("*#### 1000 nrows data only from Resume.csv*")
st.image("https://img.freepik.com/free-vector/gradient-technology-background_23-2149436181.jpg?w=1380&t=st=1701628471~exp=1701629071~hmac=3b1f0e2a5a4f3f4e3")
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
            OPENAI_API_KEY = OPENAI_API_KEY.strip()
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

# Ensure API key is available to SDKs even when provided via secrets
if OPENAI_API_KEY:
    OPENAI_API_KEY = OPENAI_API_KEY.strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
    # Read API key from env; avoid passing api_key to satisfy type-checkers across versions
    embeddings = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    # Prefer client-based construction for version robustness
    try:
        store = Qdrant.from_texts(
            texts=chunks,
            embedding=embeddings,
            client=client,
            collection_name=QDRANT_COLLECTION,
        )
        return store
    except Exception:
        # Fallback: construct wrapper then add_texts
        vector = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)
        vector.add_texts(chunks)
        return vector



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
            llm_diag = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # type: ignore[call-arg]
            _ = llm_diag.invoke("ping")
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
                qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
                emb = OpenAIEmbeddings(model=MODEL_EMBEDDING)
                vector_store = Qdrant(client=qc, collection_name=QDRANT_COLLECTION, embeddings=emb)
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

# Retrieval helper resilient to API changes with debug recording
def _record_fetch_debug(strategy: str):
    try:
        st.session_state["_debug_fetch"] = {
            "strategy": strategy,
            "retriever_cls": f"{retriever.__class__.__module__}.{retriever.__class__.__name__}",
            "has_invoke": hasattr(retriever, "invoke"),
            "has_get_relevant_documents": hasattr(retriever, "get_relevant_documents"),
            "callable": callable(retriever),
            "vector_store_cls": f"{vector_store.__class__.__module__}.{vector_store.__class__.__name__}",
            "top_k": TOP_K,
            "collection": QDRANT_COLLECTION,
        }
    except Exception:
        pass

def fetch_docs(query: str) -> List[Document]:
    # Try preferred APIs first, record which one was used
    try:
        docs = retriever.invoke(query)  # type: ignore[attr-defined]
        _record_fetch_debug("retriever.invoke")
        return docs
    except Exception:
        try:
            docs = retriever.get_relevant_documents(query)  # type: ignore[attr-defined]
            _record_fetch_debug("retriever.get_relevant_documents")
            return docs
        except Exception:
            try:
                docs = retriever(query)  # type: ignore[misc]
                _record_fetch_debug("retriever(query)")
                return docs
            except Exception:
                try:
                    docs = vector_store.similarity_search(query, k=TOP_K)  # type: ignore[attr-defined]
                    _record_fetch_debug("vector_store.similarity_search")
                    return docs
                except Exception:
                    _record_fetch_debug("none")
                    return []

# ---- RAG Chain (robust to LangChain versions) ----
llm = ChatOpenAI(model=MODEL_CHAT, temperature=0)  # type: ignore[call-arg]
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the provided context to answer the question.\n\n{context}\n\nQuestion: {input}"
)

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs or [])

# Preferred: use official helpers when available
rag_chain: Any = None
if create_stuff_documents_chain and create_retrieval_chain:
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
else:
    # Fallback: compose retrieval -> formatting -> prompt -> llm directly
    if RunnableLambda and RunnablePassthrough:
        chain = {
            "context": retriever | RunnableLambda(_format_docs),
            "input": RunnablePassthrough(),
        } | prompt | llm
        if StrOutputParser:
            chain = chain | StrOutputParser()
        rag_chain = chain
    else:
        # Last-resort simple callable; retriever returns List[Document]
        def _rag_fallback(inputs: dict):
            question = inputs.get("input", "")
            docs = fetch_docs(question)
            context = _format_docs(docs)
            msg = prompt.format_messages(context=context, input=question)
            resp = llm.invoke(msg)
            try:
                return resp.content  # type: ignore[attr-defined]
            except Exception:
                return str(resp)
        rag_chain = _rag_fallback

# ---- Tools ----
@tool
def current_datetime(_=None) -> str:
    """Return the current date and time (YYYY-MM-DD HH:MM:SS)."""
    from datetime import datetime as _dt
    return _dt.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_resumes(query: str) -> str:
    """Retrieve top resume chunks relevant to the query."""
    try:
        docs = fetch_docs(query)
        text = "\n\n".join([d.page_content for d in docs[:5]])
        # Keep output concise to reduce looping
        return text[:4000]
    except Exception as e:
        return f"Error searching resumes: {e}"

@tool("result_retriever")
def result_retriever(query: str) -> str:
    """Return retrieval results (top-k chunks) as JSON for inspection in the UI."""
    try:
        docs = fetch_docs(query)
        items = []
        for d in docs[:TOP_K]:
            meta = getattr(d, "metadata", {}) or {}
            items.append({
                "snippet": getattr(d, "page_content", ""),
                "metadata": meta,
            })
        return json.dumps({"query": query, "k": TOP_K, "results": items}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@tool
def answer_from_resumes(question: str) -> str:
    """Answer a question using the resume knowledge base via RAG."""
    try:
        # Always fetch top-k docs so tool output includes retrieval result
        docs = fetch_docs(question)
        retrieved = []
        for d in docs[:TOP_K]:
            retrieved.append({
                "snippet": getattr(d, "page_content", "")[:500],
                "metadata": getattr(d, "metadata", {}) or {},
            })

        # rag_chain may be a Runnable (with .invoke) or a plain callable depending on fallback
        try:
            result = rag_chain.invoke({"input": question})  # type: ignore[attr-defined]
        except Exception:
            result = rag_chain({"input": question})  # type: ignore[call-arg]
        # Normalize outputs across versions
        if isinstance(result, dict):
            answer = (
                result.get("answer")
                or result.get("output")
                or result.get("result")
                or str(result)
            )
        else:
            answer = str(result)

        payload = {
            "answer": str(answer)[:4000],
            "k": TOP_K,
            "retrieval": retrieved,
        }
        # Include strategy if available
        dbg = st.session_state.get("_debug_fetch")
        if isinstance(dbg, dict) and dbg.get("strategy"):
            payload["strategy"] = dbg.get("strategy")
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error answering from resumes: {e}"

# ---- Agent ----
agent_llm = ChatOpenAI(model=MODEL_CHAT, temperature=0)  # type: ignore[call-arg]
agent = create_react_agent(
    model=agent_llm,
    tools=[current_datetime, result_retriever, answer_from_resumes],
    prompt=SystemMessage(
        content=(
            "🤖🧠 You are a helpful Resume Assistant. Use tools only when needed to answer the user. Use emojis when helpful. "
            "Use 'result_retriever' to preview top retrieved chunks when helpful. "
            "Prefer 'answer_from_resumes' to synthesize a final answer from retrieved context. "
            "If you already have enough information, respond directly. After you provide an answer, stop. "
            "Be concise and include short citations when helpful. Do not call tools repeatedly."
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
if "tool_logs" not in st.session_state:
    # Each entry corresponds to one user turn: list of {name, args, output}
    st.session_state.tool_logs = []

# ---- Utilities to extract tool call details from agent messages ----
def _safe_get(obj, attr, default=None):
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

def _to_dict(obj: Any) -> Dict[str, Any]:
    # Try Pydantic model_dump; else fall back to __dict__ or str
    for key in ("model_dump", "dict"):
        fn = getattr(obj, key, None)
        if callable(fn):
            try:
                data = fn()
                return data if isinstance(data, dict) else {"value": data}
            except Exception:
                pass
    try:
        return obj.__dict__  # type: ignore[attr-defined]
    except Exception:
        return {"repr": str(obj)}

# tool call extraction disabled for now

# def extract_tool_calls(messages: List[object]):
#     """Return a list of tool call records [{name, args, output}] for a run.

#     Works across LangGraph/LangChain message versions by checking common fields.
#     """
#     calls = []
#     pending = []  # temporarily store (id,name,args) until ToolMessage arrives
#     for m in messages or []:
#         mtype = _safe_get(m, "type") or _safe_get(m, "__class__", type("x", (), {}))
#         tool_calls = _safe_get(m, "tool_calls") or _safe_get(m, "additional_kwargs", {}).get("tool_calls")
#         # AI message with tool_calls
#         if tool_calls:
#             try:
#                 for tc in tool_calls:
#                     # OpenAI shape: {id, type, function:{name, arguments}}
#                     fn = tc.get("function", {}) if isinstance(tc, dict) else {}
#                     name = fn.get("name")
#                     raw_args = fn.get("arguments")
#                     # arguments may be JSON string
#                     try:
#                         args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
#                     except Exception:
#                         args = raw_args
#                     pending.append({
#                         "id": tc.get("id") if isinstance(tc, dict) else None,
#                         "name": name,
#                         "args": args,
#                         "output": None,
#                     })
#             except Exception:
#                 pass
#             continue

#         # Tool result messages can be ToolMessage with tool_call_id
#         tool_call_id = _safe_get(m, "tool_call_id") or _to_dict(m).get("tool_call_id")
#         if tool_call_id:
#             content = _safe_get(m, "content")
#             # Match with pending
#             matched = False
#             for rec in pending:
#                 if rec.get("id") == tool_call_id:
#                     rec["output"] = content
#                     calls.append(rec)
#                     matched = True
#                     break
#             if not matched:
#                 # Unknown id; still record
#                 calls.append({"id": tool_call_id, "name": _safe_get(m, "name"), "args": None, "output": content})
#     # If some pending without outputs, still include them
#     for rec in pending:
#         if rec not in calls:
#             calls.append(rec)
#     # Trim long outputs for UI
#     for rec in calls:
#         out = rec.get("output")
#         if isinstance(out, str) and len(out) > 500:
#             rec["output"] = out[:500] + "\n... (truncated)"
#     return calls
# ---- Render existing messages ----

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return embedder.encode(text).tolist()
def search_qdrant(query_text, collection="resumes", top_k=5):
    vector = embed(query_text)
    results = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k
    )
    return results



#tool call new - --- IGNORE ---


# Assume messages is already available

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

if user_q := st.chat_input("Ask about the resumes (skills, roles, years, etc.)"):
    with st.chat_message("Human"):
        st.markdown(user_q)
    st.session_state.messages.append({"role": "Human", "content": user_q})

    with st.chat_message("AI"):
        # Constrain recursion to avoid infinite loops
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_q)]},
            config={"recursion_limit": 500},
        )
        answer = response["messages"][-1].content
        st.markdown(answer)
    st.session_state.messages.append({"role": "AI", "content": answer})
    # Collect tool call info for this turn
    try:
        tool_recs = extract_tool_calls(response.get("messages", []))
    except Exception:
        tool_recs = []
    # If no retrieval is present in tool outputs, add an automatic retrieval record
    try:
        used_tool_names = {str((rec or {}).get("name", "")).lower() for rec in tool_recs}
        if not any(name in used_tool_names for name in ("result_retriever", "answer_from_resumes")):
            auto_docs = fetch_docs(user_q)
            items = []
            for d in auto_docs[:TOP_K]:
                items.append({
                    "snippet": getattr(d, "page_content", "")[:500],
                    "metadata": getattr(d, "metadata", {}) or {},
                })
            tool_recs.append({
                "name": "result_retriever(auto)",
                "args": {"query": user_q, "k": TOP_K},
                "output": json.dumps({"results": items}, ensure_ascii=False, indent=2),
            })
    except Exception:
        pass
    st.session_state.tool_logs.append({
        "question": user_q,
        "tools": tool_recs,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

# ---- Chat controls / export ----
with st.sidebar:
    with st.expander("Chat history 🗂️", expanded=False):
        if st.session_state.get("messages"):
            for i, m in enumerate(st.session_state.messages, 1):
                role = m.get("role", "AI")
                icon = "👤" if role.lower().startswith("human") else "🤖"
                st.markdown(f"{icon} **{role} {i}:** {m.get('content','')}")

# with st.sidebar.expander("🔍 Qdrant Search Results", expanded=False):
#     hits = st.session_state.get("qdrant_hits", [])

#     if not hits:
#         st.caption("No search results available.")
#     else:
#         for i, hit in enumerate(hits):
#             st.markdown(f"**Result {i+1}**")
#             st.write(hit.payload.get("text", "No text found"))
#             st.write(f"Score: {hit.score:.4f}")
#             st.divider()


# Tool call details disabled for now
    with st.expander("Tool calls 🛠️", expanded=False):
        logs = st.session_state.get("tool_logs", [])
        if not logs:
            st.caption("No tool calls yet.")
        else:
            options = [
                f"Turn {i+1} @ {r.get('time', '')}: {r.get('question', '')[:40]}…"
                for i, r in enumerate(logs)
            ]
            sel = st.selectbox("Select a turn", options, index=len(options)-1, key="tool_calls_turn_selector")
            run = logs[options.index(sel)]
            tools = run.get("tools", [])

            if not tools:
                st.caption("No tools used")
            else:
                for i, tool in enumerate(tools, 1):
                    st.markdown(f"**{i}. {tool.get('name', '<unknown>')}**")
                    st.code(json.dumps(tool.get("args"), ensure_ascii=False, indent=2) if isinstance(tool.get("args"), (dict, list)) else str(tool.get("args")))
                    if tool.get("output"):
                        st.caption("Output")
                        st.code(tool["output"])


    # with st.expander("Tool calls 🛠️", expanded=False):
    #     logs = st.session_state.get("tool_logs") or []
    #     if logs:
    #         options = [f"Turn {i+1} @ {r.get('time','')}: {r.get('question','')[:40]}…" for i, r in enumerate(logs)]
    #         sel = st.selectbox("Select a turn", options=options, index=len(options)-1, key="tool_calls_turn_selector")
    #         sel_idx = options.index(sel) if sel in options else len(options)-1
    #         run = logs[sel_idx]
    #         tools = run.get("tools", [])
    #         if not tools:
    #             st.caption("No tools used")
    #         for j, rec in enumerate(tools, 1):
    #             name = rec.get("name") or "<unknown>"
    #             args = rec.get("args")
    #             out = rec.get("output")
    #             st.markdown(f"**{j}. {name}**")
    #             st.code(json.dumps(args, ensure_ascii=False, indent=2) if isinstance(args, (dict, list)) else str(args))
    #             if out:
    #                 st.caption("Output")
    #                 st.code(out)
    #     else:
    #         st.caption("No tool calls yet.")
        st.divider()

        st.subheader("Chat export")
        has_msgs = bool(st.session_state.get("messages"))
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        def _to_markdown(msgs):
            parts = []
            for m in msgs:
                role = m.get("role", "AI")
                content = str(m.get("content", "")).strip()
                parts.append(f"### {role}\n\n{content}\n\n---")
            return "\n".join(parts)

    if has_msgs:
        md = _to_markdown(st.session_state.messages)
        js = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            label="Download chat (Markdown)",
            data=md,
            file_name=f"chat-{ts}.md",
            mime="text/markdown",
        )
        st.download_button(
            label="Download chat (JSON)",
            data=js,
            file_name=f"chat-{ts}.json",
            mime="application/json",
        )
    else:
        st.caption("No messages yet — ask something to start the chat.")

    if st.button("Clear chat history", type="secondary", disabled=not has_msgs):
        st.session_state.messages = []
        # Streamlit 1.32+: st.rerun() is stable; older versions use experimental_rerun
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            _er = getattr(st, "experimental_rerun", None)
            if callable(_er):
                _er()
