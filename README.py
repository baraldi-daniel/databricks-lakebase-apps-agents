# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Semantic Cache — Complete Replication Guide
# MAGIC
# MAGIC Sales agent with **LLM semantic cache**, **guardrails via LLM-as-a-Judge** (evaluated with MLflow Evaluation Runs),
# MAGIC and **intelligent routing via LangGraph**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## How to use this guide
# MAGIC
# MAGIC Run the notebooks **in order**. Each one is self-contained and executable.
# MAGIC
# MAGIC | Step | Notebook | What it does | Est. time |
# MAGIC |------|----------|--------------|-----------|
# MAGIC | 1 | `01_setup_lakebase` | Creates tables, user and sample products in Lakebase | 2 min |
# MAGIC | 2 | `02_source_code` | Generates all 7 app files in the workspace | 1 min |
# MAGIC | 3 | `03_deploy` | Updates app.yaml, creates and deploys the app | 3 min |
# MAGIC | 4 | `04_register_judges` | Registers guardrails as judges in MLflow | 1 min |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC ### 1. Databricks Workspace
# MAGIC - Workspace with **Lakebase** support (FEVM/serverless)
# MAGIC - Databricks CLI v0.229+ authenticated (`databricks auth login`)
# MAGIC
# MAGIC ### 2. Create a Lakebase instance
# MAGIC 1. In the workspace, go to **SQL > Lakebase** (sidebar)
# MAGIC 2. Click **Create** to provision an instance
# MAGIC 3. Wait until status is **Running**
# MAGIC 4. Copy the **Host** (e.g. `ep-xxx.database.us-west-2.cloud.databricks.com`)
# MAGIC    - Click the instance > copy the host shown
# MAGIC
# MAGIC ### 3. Where to find credentials
# MAGIC
# MAGIC | Credential | Where to find it |
# MAGIC |------------|------------------|
# MAGIC | `LAKEBASE_HOST` | SQL > Lakebase > your instance > **Host** field |
# MAGIC | `LAKEBASE_PG_PASSWORD` | You define it when creating the user in notebook `01_setup_lakebase` (choose a strong password) |
# MAGIC | `WORKSPACE_URL` | Your workspace URL (e.g. `https://my-workspace.cloud.databricks.com`) |
# MAGIC | `DATABRICKS_APP_URL` | Auto-generated when creating the app in step 3 |
# MAGIC | `USER_EMAIL` | Your Databricks workspace email |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Tech Stack
# MAGIC - **Frontend**: Streamlit
# MAGIC - **Database**: Lakebase (managed PostgreSQL)
# MAGIC - **Main LLM**: Claude Sonnet 4 (via Foundation Model API)
# MAGIC - **Cache analysis**: Llama 3.3 70B (fast/cheap for cacheability decisions)
# MAGIC - **Embeddings**: GTE Large EN (for semantic similarity)
# MAGIC - **Cache orchestration**: LangGraph
# MAGIC - **Guardrail evaluation**: MLflow 3 GenAI Evaluation (LLM-as-a-Judge)
# MAGIC
# MAGIC ## Architecture
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────┐
# MAGIC │                    Streamlit UI (app.py)                │
# MAGIC │  ┌─────────────────────┐  ┌──────────────────────────┐ │
# MAGIC │  │  Without Cache       │  │  With Semantic Cache      │ │
# MAGIC │  │  (direct LLM)        │  │  (LangGraph routing)      │ │
# MAGIC │  └─────────┬───────────┘  └──────────┬───────────────┘ │
# MAGIC └────────────┼─────────────────────────┼─────────────────┘
# MAGIC              │                         │
# MAGIC              │                ┌────────▼─────────┐
# MAGIC              │                │  cache_graph.py   │
# MAGIC              │                │  Llama 3.3 70B    │
# MAGIC              │                │  analyzes if      │
# MAGIC              │                │  cacheable         │
# MAGIC              │                └────────┬─────────┘
# MAGIC              │                         │
# MAGIC              │     cache miss? ────────┤
# MAGIC              │     reuses same call     │ cache hit?
# MAGIC              │                         │ returns instantly
# MAGIC     ┌────────▼─────────────────────────▼───────┐
# MAGIC     │           agent.py — call_llm()           │
# MAGIC     │  Claude Sonnet 4 + Tool Calling           │
# MAGIC     │  (sales, products)                        │
# MAGIC     └────────┬─────────────────────────────────┘
# MAGIC              │ (separate thread)
# MAGIC     ┌────────▼─────────────────────────────────┐
# MAGIC     │        guardrail_eval.py                  │
# MAGIC     │  LLM-as-a-Judge → MLflow Eval Run        │
# MAGIC     │  + saves to Lakebase                      │
# MAGIC     └──────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Application Files
# MAGIC
# MAGIC | File | Description |
# MAGIC |------|-------------|
# MAGIC | `app.py` | Streamlit UI — 3 tabs (comparison, history, settings) |
# MAGIC | `agent.py` | Agent with Claude Sonnet 4 + tool calling |
# MAGIC | `cache_graph.py` | LangGraph: analyzes cacheability with Llama 3.3 70B |
# MAGIC | `db.py` | Lakebase access — CRUD, 2-tier cache, guardrails |
# MAGIC | `guardrail_eval.py` | LLM-as-a-Judge + MLflow Evaluation Runs |
# MAGIC | `app.yaml` | Databricks App config (env vars, command) |
# MAGIC | `requirements.txt` | Python dependencies |
# MAGIC
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC | Problem | Cause | Solution |
# MAGIC |---------|-------|----------|
# MAGIC | App Not Available | App not listening on port 8000 | Check `command` in `app.yaml` |
# MAGIC | Lakebase connection error | Wrong credentials | Check env vars in `app.yaml` |
# MAGIC | `InsufficientPrivilege` | User is not table owner | Connect as owner and run `GRANT ALL` |
# MAGIC | `UndefinedColumn: name` | Missing column in guardrails table | Re-run notebook `01_setup_lakebase` |
# MAGIC | Judges not showing in MLflow | Missing `.start()` after `.register()` | Run notebook `04_register_judges` |
# MAGIC | Guardrail eval fails | Expired MLflow token | Check logs at `<URL>/logz` |
# MAGIC
# MAGIC **App logs**: Go to `<APP_URL>/logz` for real-time logs.