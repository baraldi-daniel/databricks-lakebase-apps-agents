# LLM Semantic Cache + AI Sales Agent on Databricks

An AI-powered sales agent with two-tier semantic caching, configurable guardrails (LLM-as-a-Judge), full MLflow observability, and tool calling — deployed as a Databricks App with Lakebase (managed PostgreSQL).

## Features

- **AI Sales Agent** — Natural language conversations with tool calling (list products, register sales, suggest items) powered by Claude Sonnet 4 via Databricks Foundation Model API
- **Two-Tier Semantic Cache** — Exact hash match + embedding similarity (cosine >= 0.92) to avoid redundant LLM calls, with smart cacheability detection via LLM judge
- **Configurable Guardrails** — Business rules in natural language, stored in the database and evaluated automatically after every response using LLM-as-a-Judge
- **MLflow Observability** — Automatic tracing of all LLM calls, Evaluation Runs with per-rule pass/fail metrics, and registered Judges for continuous monitoring
- **Sales Dashboard** — KPIs (revenue, total sales, items sold), charts by product, and stock tracking
- **Lakebase Backend** — All data (products, sales, cache, conversations, guardrails) persisted in Databricks' managed PostgreSQL

## Architecture

<img width="739" height="581" alt="image" src="https://github.com/user-attachments/assets/96ce4ae5-9364-4dfa-8ecd-75bc44d281fb" />


## Project Structure

```
llm-cache-guide/
├── 01_setup_lakebase.py          # Notebook: create tables, seed data, configure permissions
├── 02_codigo_fonte.py            # Notebook: generate all application source files
├── 03_deploy.py                  # Notebook: create and deploy the Databricks App
├── 04_registrar_judges.py        # Notebook: register guardrails as MLflow Judges
├── README.py                     # Notebook: overview and instructions
│
├── src_app.py                    # Streamlit UI (4 tabs)
├── src_agent.py                  # LLM agent with tool calling + guardrail eval
├── src_db.py                     # Lakebase connection, cache, CRUD operations
├── src_guardrail_eval.py         # LLM-as-a-Judge + MLflow Evaluation Runs
├── src_cache_graph.py            # Cache decision graph (LangGraph)
├── src_app.yaml                  # Databricks App configuration
├── src_requirements.txt          # Python dependencies
│
└── codigo-fonte/                 # Standalone source files (alternative to notebooks)
    ├── app.py
    ├── agent.py
    ├── db.py
    ├── guardrail_eval.py
    ├── cache_graph.py
    ├── app.yaml
    └── requirements.txt
```

## Prerequisites

- Databricks workspace with **Lakebase** enabled
- **SQL Warehouse** (Serverless Starter works)
- **Foundation Model API** enabled (Claude Sonnet 4 endpoint)
- Databricks CLI v0.229.0+ authenticated

## Quick Start

### 1. Setup Lakebase

Run notebook `01_setup_lakebase`. Fill in your Lakebase connection string and password. This creates:

- `products` — store catalog with seed data (laptops, phones, accessories)
- `sales` — transactional records
- `llm_cache` — semantic cache with embeddings
- `conversations` — chat history
- `guardrails` — configurable business rules

### 2. Generate Source Code

Run notebook `02_codigo_fonte`. This copies the `src_*.py` files into your app directory at `/Workspace/Users/<email>/llm-cache-app/`.

### 3. Deploy

Run notebook `03_deploy`. Fill in your app name, Lakebase connection, and password. This:

- Creates the `app.yaml` with your credentials
- Creates the Databricks App via SDK
- Sets up the MLflow experiment and grants Service Principal permissions
- Deploys the application

After deployment, add a **SQL Warehouse** resource to the app via the UI (Compute > Apps > your-app > Settings).

### 4. Register Judges (Optional)

Run notebook `04_registrar_judges` to register existing guardrail rules as MLflow Judges. The app UI also registers judges automatically when you create rules via the Settings tab.

## How It Works

### Semantic Cache

Not every query should be cached. An LLM judge classifies each message as `CACHEABLE` (generic questions) or `NOT_CACHEABLE` (personal/context-dependent messages).

For cacheable queries, the system checks two tiers:
1. **Exact match** — MD5 hash comparison (instant, zero cost)
2. **Semantic match** — Embedding similarity via `databricks-gte-large-en` (cosine >= 0.92)

### Tool Calling

The agent has 3 tools: `list_products`, `create_sale`, and `list_sales`. When a sale is registered, it's saved to Lakebase and also logged to a Unity Catalog table for visibility in Data Explorer.

### Guardrail Evaluation

After every agent response, a judge LLM evaluates compliance with all active rules. Results are logged as MLflow Evaluation Runs with per-rule Feedback objects and displayed in the app UI.

### MLflow Integration

- **Autolog traces** — Every LLM call is traced automatically
- **Evaluation Runs** — Guardrail assessments with pass/fail metrics
- **Registered Judges** — Guardrails as Guidelines scorers for continuous monitoring

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Database | Lakebase (managed PostgreSQL) |
| LLM | Claude Sonnet 4 (Databricks Foundation Model API) |
| Embeddings | databricks-gte-large-en |
| Observability | MLflow 3 (Traces + Evaluation Runs + Judges) |
| Agent Framework | LangGraph |
| Deployment | Databricks Apps |

## License

MIT
