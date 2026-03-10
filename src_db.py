import os
import hashlib
import psycopg
import json
import re
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

NOT_CACHEABLE = specific/personal message containing names, emails, order numbers, quantities, confirmations, or context-dependent replies (e.g. "register a sale for João", "sim", "quero 3 iphones", "meu pedido #123", "ok fechar")

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
