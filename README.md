# Resume Assistant (Qdrant RAG + ReAct Agent)

A Streamlit app that answers questions about a resume dataset using a Qdrant vector database, OpenAI embeddings, and a ReAct agent.

## Features
- Upload/Load `Resume.csv` and index text chunks in Qdrant
- Retrieve relevant resume snippets with RAG
- ReAct agent (gpt-4o-mini) uses tools to answer questions
- Diagnostics panel to verify OpenAI and Qdrant connectivity

## Local Setup

1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables or create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   QDRANT_URL=https://YOUR-CLUSTER.qdrant.tech
   QDRANT_API_KEY=your_qdrant_key
   ```
4. Place `Resume.csv` next to the app (must include a `Resume` column or `Resume_str` + `Category`).
5. Run Streamlit:
   ```bash
   streamlit run Capstone3_RAG.py
   ```

## Deploy to Streamlit Cloud

1. Push this folder to GitHub (include `Capstone3_RAG.py`, `requirements.txt`, and `Resume.csv` or wire in your own data source).
2. On streamlit.io, create a new app and point it to your repo and `Capstone3_RAG.py`.
3. Set Secrets under Settings → Secrets:
   ```toml
   OPENAI_API_KEY = "your_openai_key"
   QDRANT_URL = "https://YOUR-CLUSTER.qdrant.tech"
   QDRANT_API_KEY = "your_qdrant_key"
   ```
4. Deploy. The app will read secrets first, then env vars, then prompt for any missing values.

## Tips
- First run will index in Qdrant; subsequent runs may be faster if the collection is reused.
- Adjust chunk size/overlap in `build_qdrant_store` for your data size.
- For large datasets, consider batching and idempotent collection management.

## Troubleshooting
- Use the Diagnostics panel in the app to verify:
  - OpenAI Embeddings and Chat access
  - Qdrant connectivity and collections listing
- Common issues:
  - Invalid keys → Update Streamlit Secrets or .env
  - Wrong Qdrant URL → Must be full https URL for your cluster
  - Model access → Ensure your account has access to `gpt-4o-mini` and `text-embedding-3-small`
