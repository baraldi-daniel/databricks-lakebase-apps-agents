# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2: Application Source Code
# MAGIC
# MAGIC This notebook contains **all** the Python files that make up the application.
# MAGIC Each cell creates a file in `/Workspace/Users/<your-email>/llm-cache-app/`.
# MAGIC
# MAGIC **Important**: Your email is detected automatically. Just run all cells.

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient

# Auto-detect user email from workspace context
USER_EMAIL = WorkspaceClient().current_user.me().user_name
APP_DIR = f"/Workspace/Users/{USER_EMAIL}/llm-cache-app"

os.makedirs(APP_DIR, exist_ok=True)
print(f"Directory: {APP_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### requirements.txt

# COMMAND ----------

with open(f"{APP_DIR}/requirements.txt", "w") as f:
    f.write("""streamlit==1.45.1
psycopg[binary]==3.2.9
databricks-sdk>=0.55.0
pandas>=2.0.0
openai>=1.0.0
mlflow[databricks]>=3.1.0
langgraph>=0.2.0
langchain-core>=0.3.0
""")
print("requirements.txt created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### app.yaml
# MAGIC
# MAGIC **Fill in LAKEBASE_HOST, password and app URL values before deploying.**
# MAGIC The app URL will only be known after creating the app in Step 3.

# COMMAND ----------

with open(f"{APP_DIR}/app.yaml", "w") as f:
    f.write("""command:
  - streamlit
  - run
  - app.py
  - --server.port
  - "8000"
  - --server.address
  - "0.0.0.0"

env:
  - name: DATABRICKS_HOST
    value: "PREENCHA_COM_URL_DO_WORKSPACE"
  - name: LAKEBASE_HOST
    value: "PREENCHA_COM_HOST_LAKEBASE"
  - name: LAKEBASE_PG_USER
    value: "cache_app"
  - name: LAKEBASE_PG_PASSWORD
    value: "PREENCHA_COM_SENHA"
  - name: DATABRICKS_APP_URL
    value: "PREENCHA_APOS_CRIAR_APP"
  - name: APP_OWNER_EMAIL
    value: "PREENCHA_COM_SEU_EMAIL"

resources:
  - name: sql-warehouse
    sql_warehouse:
      permission: CAN_USE
""")
print("app.yaml created (FILL IN the values before deploying!)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### db.py — Database access (Lakebase)
# MAGIC
# MAGIC Contains: connection, products/sales CRUD, 2-tier semantic cache
# MAGIC (MD5 hash + cosine similarity), conversations, guardrails.

# COMMAND ----------

with open(f"{APP_DIR}/db.py", "w") as f:
    f.write('''import os
import hashlib
import psycopg
import json
import math


def get_connection():
    return psycopg.connect(
        host=os.environ["LAKEBASE_HOST"],
        port=5432,
        dbname="databricks_postgres",
        user=os.environ["LAKEBASE_PG_USER"],
        password=os.environ["LAKEBASE_PG_PASSWORD"],
        sslmode="require",
        autocommit=True,
    )


def get_products(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM products ORDER BY category, name")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return rows


def get_sales(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, p.name as product_name, p.category, p.sku
        FROM sales s JOIN products p ON s.product_id = p.id
        ORDER BY s.sale_date DESC LIMIT 100
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return rows


def create_sale(conn, product_id, customer_name, quantity):
    cur = conn.cursor()
    cur.execute("SELECT price, stock FROM products WHERE id = %s", (product_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        return None, "Produto nao encontrado"
    price, stock = row
    if quantity > stock:
        cur.close()
        return None, f"Estoque insuficiente ({stock} disponiveis)"
    total = float(price) * quantity
    cur.execute("""
        INSERT INTO sales (product_id, customer_name, quantity, unit_price, total)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
    """, (product_id, customer_name, quantity, price, total))
    sale_id = cur.fetchone()[0]
    cur.execute("UPDATE products SET stock = stock - %s WHERE id = %s", (quantity, product_id))
    cur.close()
    return sale_id, f"Venda #{sale_id} registrada! {quantity}x por R$ {total:,.2f}"


# ---- Hash + Cosine Similarity ----

def _md5(text):
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---- Smart Cache: LLM-as-a-judge for cacheability ----

def is_cacheable_query(query_text):
    """Use LLM to decide if a query is generic (cacheable) or specific (not cacheable).
    Only cache longer, self-contained questions. Short/contextual messages skip cache."""
    text = query_text.strip()

    # Too short — always skip (save LLM call)
    if len(text) < 10:
        return False

    try:
        from openai import OpenAI
        host = os.environ.get("DATABRICKS_HOST", "")
        if host and not host.startswith("http"):
            host = f"https://{host}"
        token = os.environ.get("DATABRICKS_TOKEN", "")
        if not token:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            auth = w.config.authenticate()
            h = {}
            if callable(auth):
                auth(h)
            else:
                h = auth
            token = h.get("Authorization", "").replace("Bearer ", "")
        client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

        response = client.chat.completions.create(
            model=os.environ.get("SERVING_ENDPOINT", "databricks-claude-sonnet-4"),
            messages=[{
                "role": "user",
                "content": f"""Classify this user message as CACHEABLE or NOT_CACHEABLE.

CACHEABLE = generic question that anyone could ask and get the same answer (e.g. "what products do you have?", "what are the categories?", "how does the return policy work?")

NOT_CACHEABLE = specific/personal message containing names, emails, order numbers, quantities, confirmations, or context-dependent replies (e.g. "register a sale for Joao", "sim", "quero 3 iphones", "meu pedido #123", "ok fechar")

Message: "{text}"

Reply with ONLY one word: CACHEABLE or NOT_CACHEABLE"""
            }],
            max_tokens=5,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().upper()
        return "CACHEABLE" in answer and "NOT" not in answer
    except Exception:
        # If LLM judge fails, default to not cacheable (safer)
        return False


# ---- Two-Tier Semantic LLM Cache ----

SIMILARITY_THRESHOLD = 0.92


def cache_get_exact(conn, query_text):
    """Tier 1: Exact hash match. No embedding API call needed."""
    query_hash = _md5(query_text)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, query_text, response_text, response_time_ms FROM llm_cache WHERE query_hash = %s LIMIT 1",
        (query_hash,),
    )
    row = cur.fetchone()
    if row:
        rid, cached_query, cached_resp, resp_time = row
        cur.execute("UPDATE llm_cache SET hit_count = hit_count + 1, last_hit_at = NOW() WHERE id = %s", (rid,))
        cur.close()
        return cached_resp, resp_time, 1.0, cached_query
    cur.close()
    return None, 0, 0.0, None


def cache_get_semantic(conn, query_text, query_embedding):
    """Tier 2: Semantic similarity match using embeddings."""
    cur = conn.cursor()
    cur.execute("SELECT id, query_text, response_text, embedding, response_time_ms FROM llm_cache")
    rows = cur.fetchall()

    best_id = None
    best_sim = 0.0
    best_response = None
    best_query = None
    best_time = 0

    for row in rows:
        rid, cached_query, cached_resp, emb_json, resp_time = row
        if not emb_json:
            continue
        cached_emb = json.loads(emb_json)
        sim = _cosine_similarity(query_embedding, cached_emb)
        if sim > best_sim:
            best_sim = sim
            best_id = rid
            best_response = cached_resp
            best_query = cached_query
            best_time = resp_time

    if best_sim >= SIMILARITY_THRESHOLD and best_id:
        cur.execute("UPDATE llm_cache SET hit_count = hit_count + 1, last_hit_at = NOW() WHERE id = %s", (best_id,))
        cur.close()
        return best_response, best_time, best_sim, best_query
    cur.close()
    return None, 0, best_sim, None


def cache_set(conn, query_text, response_text, embedding, model, tokens_used, response_time_ms):
    cur = conn.cursor()
    emb_json = json.dumps(embedding) if embedding else None
    query_hash = _md5(query_text)
    cur.execute("""
        INSERT INTO llm_cache (query_text, response_text, embedding, model, tokens_used, response_time_ms, query_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (query_text, response_text, emb_json, model, tokens_used, response_time_ms, query_hash))
    cur.close()


def get_cache_stats(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as total_entries,
               COALESCE(SUM(hit_count), 0) as total_hits,
               COALESCE(ROUND(AVG(response_time_ms)), 0) as avg_response_ms
        FROM llm_cache
    """)
    row = cur.fetchone()
    cur.close()
    return {"entries": row[0] or 0, "hits": int(row[1] or 0), "avg_ms": int(row[2] or 0)}


def get_cache_entries(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT query_text, response_text, hit_count, response_time_ms, created_at, last_hit_at
        FROM llm_cache ORDER BY hit_count DESC
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return rows


# ---- Conversations ----

def save_message(conn, session_id, role, content, cached=False, response_time_ms=0, cache_hit=False, similarity=0.0):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversations (session_id, role, content, cached, response_time_ms, cache_hit, similarity)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (session_id, role, content, cached, response_time_ms, cache_hit, similarity))
    cur.close()


def get_conversations(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT session_id, role, content, cached, response_time_ms, cache_hit,
               COALESCE(similarity, 0) as similarity, created_at
        FROM conversations
        WHERE cached = TRUE
        ORDER BY created_at DESC LIMIT 200
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return rows


# ---- Guardrails ----

def get_guardrails(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, name, rule_text, enabled FROM guardrails ORDER BY id")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return rows


def add_guardrail(conn, rule_text, name=""):
    cur = conn.cursor()
    cur.execute("INSERT INTO guardrails (name, rule_text) VALUES (%s, %s) RETURNING id", (name, rule_text))
    rid = cur.fetchone()[0]
    cur.close()
    return rid


def update_guardrail(conn, guardrail_id, name=None, rule_text=None):
    cur = conn.cursor()
    if name is not None and rule_text is not None:
        cur.execute("UPDATE guardrails SET name = %s, rule_text = %s WHERE id = %s", (name, rule_text, guardrail_id))
    elif name is not None:
        cur.execute("UPDATE guardrails SET name = %s WHERE id = %s", (name, guardrail_id))
    elif rule_text is not None:
        cur.execute("UPDATE guardrails SET rule_text = %s WHERE id = %s", (rule_text, guardrail_id))
    cur.close()


def toggle_guardrail(conn, guardrail_id, enabled):
    cur = conn.cursor()
    cur.execute("UPDATE guardrails SET enabled = %s WHERE id = %s", (enabled, guardrail_id))
    cur.close()


def delete_guardrail(conn, guardrail_id):
    cur = conn.cursor()
    cur.execute("DELETE FROM guardrails WHERE id = %s", (guardrail_id,))
    cur.close()
''')
print("db.py created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### agent.py — Sales agent with tool calling
# MAGIC
# MAGIC Main LLM: Claude Sonnet 4 (via Foundation Model API).
# MAGIC Includes tool calling, MLflow tracing, and guardrail evaluation.

# COMMAND ----------

# Copy agent.py from guide source files
src = f"/Workspace/Users/{USER_EMAIL}/llm-cache-guide/src_agent.py"
dst = f"{APP_DIR}/agent.py"
with open(src, "r") as fin:
    content = fin.read()
with open(dst, "w") as fout:
    fout.write(content)
print(f"agent.py copied from {src}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### cache_graph.py - Intelligent cache routing with LangGraph

# COMMAND ----------

# MAGIC %md
# MAGIC ### cache_graph.py — Intelligent cache routing with LangGraph
# MAGIC
# MAGIC Uses Llama 3.3 70B (fast/cheap) to analyze if a message can be cached.
# MAGIC Graph: analyze -> check_cache (if cacheable) -> END

# COMMAND ----------

with open(f"{APP_DIR}/cache_graph.py", "w") as f:
    f.write('''"""LangGraph-based cache routing with LLM analysis for cacheability decisions."""
import json
import os
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from openai import OpenAI
import db


# ---- State ----

class CacheState(TypedDict):
    messages: List[dict]
    user_query: str
    is_cacheable: bool
    cache_analysis: str
    cache_result: Optional[dict]


# ---- LLM Client ----

def _get_client():
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    header_factory = w.config.authenticate()
    if callable(header_factory):
        h = {}
        header_factory(h)
        token = h.get("Authorization", "").replace("Bearer ", "")
    elif isinstance(header_factory, dict):
        token = header_factory.get("Authorization", "").replace("Bearer ", "")
    else:
        token = ""
    host = os.environ.get("DATABRICKS_HOST", "")
    if host and not host.startswith("http"):
        host = f"https://{host}"
    return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")


# ---- Nodes ----

ANALYSIS_PROMPT = """Voce e um analisador de cache para um agente de vendas de tecnologia.

Analise a conversa abaixo e a ULTIMA mensagem do usuario. Decida se a resposta a essa mensagem pode ser CACHEADA para reuso futuro.

CACHEAR quando a pergunta e:
- Generica e autocontida (ex: "quais notebooks voces tem?", "qual o preco do macbook?")
- Perguntas sobre catalogo, categorias, precos que qualquer usuario faria
- Nao depende do contexto da conversa anterior

NAO CACHEAR quando:
- A mensagem depende do contexto (ex: "sim", "quero esse", "pode ser", "e esse?")
- Contem dados pessoais (nome, email, telefone)
- E uma confirmacao ou continuacao de fluxo (compra, pedido)
- Referencia algo dito anteriormente ("o primeiro", "aquele", "esse mesmo")
- E muito curta e ambigua sem contexto

Conversa:
{conversation}

Ultima mensagem: "{last_message}"

Responda APENAS em JSON:
{{"cacheable": true/false, "reason": "explicacao curta"}}"""


def analyze_cacheability(state: CacheState) -> dict:
    """Use fast LLM to analyze if query is cacheable based on full context."""
    import re
    user_query = state["user_query"]

    if re.search(r\'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\', user_query):
        return {"is_cacheable": False, "cache_analysis": "Contem email"}
    if re.search(r\'#\\d+|venda\\s+\\d+|pedido\\s+\\d+\', user_query, re.IGNORECASE):
        return {"is_cacheable": False, "cache_analysis": "Referencia pedido/venda"}

    msgs = state["messages"]
    conv_lines = []
    for m in msgs[:-1]:
        c = m.get("content", "")
        if c:
            role = "Usuario" if m["role"] == "user" else "Agente"
            conv_lines.append(f"{role}: {c[:200]}")
    conversation = "\\n".join(conv_lines[-10:]) if conv_lines else "(primeira mensagem)"

    prompt = ANALYSIS_PROMPT.format(conversation=conversation, last_message=user_query)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="databricks-meta-llama-3-3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        return {
            "is_cacheable": result.get("cacheable", False),
            "cache_analysis": result.get("reason", ""),
        }
    except Exception as e:
        return {"is_cacheable": False, "cache_analysis": f"Erro na analise: {e}"}


def check_cache(state: CacheState) -> dict:
    """Check cache for the query."""
    import agent as agent_mod
    conn = db.get_connection()
    try:
        cached_resp, cached_time, similarity, matched_query = db.cache_get_exact(conn, state["user_query"])
        if cached_resp:
            return {"cache_result": {
                "response": cached_resp, "time_ms": 1, "hit": True,
                "similarity": similarity, "matched_query": matched_query
            }}

        query_embedding = agent_mod.get_embedding(state["user_query"])
        cached_resp, cached_time, similarity, matched_query = db.cache_get_semantic(conn, state["user_query"], query_embedding)
        if cached_resp:
            return {"cache_result": {
                "response": cached_resp, "time_ms": 1, "hit": True,
                "similarity": similarity, "matched_query": matched_query
            }}
    except Exception:
        pass
    finally:
        conn.close()
    return {"cache_result": None}


# ---- Routing ----

def route_after_analysis(state: CacheState) -> str:
    if state["is_cacheable"]:
        return "check_cache"
    return END


# ---- Build Graph ----

def build_cache_graph():
    graph = StateGraph(CacheState)

    graph.add_node("analyze", analyze_cacheability)
    graph.add_node("check_cache", check_cache)

    graph.set_entry_point("analyze")
    graph.add_conditional_edges("analyze", route_after_analysis, {
        "check_cache": "check_cache",
        END: END,
    })
    graph.add_edge("check_cache", END)

    return graph.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_cache_graph()
    return _graph


def run_cache_check(messages):
    """Run the cache analysis graph.
    Returns (cache_hit_response, time_ms, is_hit, similarity, matched_query, is_cacheable, analysis)
    or None values if no cache hit."""
    graph = get_graph()
    user_query = messages[-1]["content"]

    state = graph.invoke({
        "messages": messages,
        "user_query": user_query,
        "is_cacheable": False,
        "cache_analysis": "",
        "cache_result": None,
    })

    is_cacheable = state.get("is_cacheable", False)
    analysis = state.get("cache_analysis", "")
    cache_result = state.get("cache_result")

    if cache_result:
        return (
            cache_result["response"],
            cache_result["time_ms"],
            True,
            cache_result.get("similarity", 1.0),
            cache_result.get("matched_query"),
            is_cacheable,
            analysis,
        )

    return None, 0, False, 0.0, None, is_cacheable, analysis
''')
print("cache_graph.py created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### guardrail_eval.py — LLM-as-a-Judge + MLflow Evaluation Runs
# MAGIC
# MAGIC Evaluates agent responses against guardrail rules using LLM as judge.
# MAGIC Logs results to MLflow as Evaluation Runs and saves to Lakebase.
# MAGIC Also registers/removes judges in the MLflow Judges tab.

# COMMAND ----------

src = f"/Workspace/Users/{USER_EMAIL}/llm-cache-guide/src_guardrail_eval.py"
dst = f"{APP_DIR}/guardrail_eval.py"
with open(src, "r") as fin:
    content = fin.read()
with open(dst, "w") as fout:
    fout.write(content)
print(f"guardrail_eval.py copied from {src}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### app.py — Streamlit UI (main interface)
# MAGIC
# MAGIC 4 tabs: Side-by-side comparison (with/without cache), Conversations & Cache, Vendas (sales tracking), Settings.
# MAGIC Includes guardrail evaluations (LLM-as-a-Judge) fetched from MLflow Traces API.

# COMMAND ----------

# Copy app.py from the guide's source files
src = f"/Workspace/Users/{USER_EMAIL}/llm-cache-guide/src_app.py"
dst = f"{APP_DIR}/app.py"
with open(src, "r") as fin:
    content = fin.read()
with open(dst, "w") as fout:
    fout.write(content)
print(f"app.py copied from {src}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final verification
# MAGIC
# MAGIC Confirms all files were created correctly.

# COMMAND ----------

import os

expected = ["requirements.txt", "app.yaml", "db.py", "agent.py", "cache_graph.py", "guardrail_eval.py", "app.py"]
for f in expected:
    path = f"{APP_DIR}/{f}"
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "OK" if exists and size > 0 else "FALTANDO"
    print(f"  [{status}] {f} ({size:,} bytes)")

print(f"\nAll {len(expected)} files created in {APP_DIR}")
print("Next step: Run notebook 03_deploy to deploy the app.")