import json
import os
import time
from openai import OpenAI
import db

# Setup MLflow tracing
_mlflow_enabled = False
_mlflow_owner = os.environ.get("APP_OWNER_EMAIL", "")
try:
    import mlflow
    mlflow.set_tracking_uri("databricks")
    if _mlflow_owner:
        mlflow.set_experiment(f"/Users/{_mlflow_owner}/guardrail-evaluations")
    mlflow.openai.autolog(log_traces=True)
    _mlflow_enabled = True
    print("[agent] MLflow tracing enabled", flush=True)
except Exception as _e:
    print(f"[agent] MLflow tracing setup failed: {_e}", flush=True)

SYSTEM_PROMPT = """Voce e um agente de vendas de uma loja de tecnologia e eletronicos. Voce ajuda clientes a:
1. Consultar produtos disponiveis (nome, preco, estoque, categoria)
2. Registrar vendas (precisa do nome do cliente, produto, quantidade)
3. Sugerir produtos com base no que o cliente procura
4. Informar sobre promocoes e comparar produtos

Produtos disponiveis:
{products_info}

REGRAS:
- Sempre responda em portugues do Brasil
- Formate precos em R$
- Seja prestativo e sugira produtos complementares
- Use as funcoes disponiveis para executar acoes
- Nunca invente dados — use as funcoes
- IMPORTANTE: Para registrar uma venda, voce DEVE coletar o nome do cliente, produto e quantidade.

{guardrails}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_sale",
            "description": "Registra uma venda de produto. Exige nome do cliente, produto e quantidade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "ID do produto"},
                    "customer_name": {"type": "string", "description": "Nome do cliente"},
                    "quantity": {"type": "integer", "description": "Quantidade", "default": 1},
                },
                "required": ["product_id", "customer_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "Lista produtos disponiveis, opcionalmente filtrando por categoria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Categoria para filtrar"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sales",
            "description": "Lista as vendas mais recentes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _get_token():
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    header_factory = w.config.authenticate()
    if callable(header_factory):
        h = {}
        header_factory(h)
        return h.get("Authorization", "").replace("Bearer ", "")
    elif isinstance(header_factory, dict):
        return header_factory.get("Authorization", "").replace("Bearer ", "")
    return ""


def _get_client():
    host = os.environ.get("DATABRICKS_HOST", "")
    if host and not host.startswith("http"):
        host = f"https://{host}"
    return OpenAI(api_key=_get_token(), base_url=f"{host}/serving-endpoints")


def get_embedding(text):
    client = _get_client()
    response = client.embeddings.create(model="databricks-gte-large-en", input=text)
    return response.data[0].embedding


def _log_sale_to_uc(product_id, customer_name, quantity, sale_msg):
    """Log sale to Unity Catalog table via SQL Statement API (visible in Data Explorer)."""
    try:
        import requests
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        host = os.environ.get("DATABRICKS_HOST", "")
        if host and not host.startswith("http"):
            host = f"https://{host}"
        header_factory = w.config.authenticate()
        headers = {}
        if callable(header_factory):
            header_factory(headers)
        elif isinstance(header_factory, dict):
            headers = header_factory

        catalog = os.environ.get("UC_CATALOG", "baraldiworkspace_catalog")
        schema = os.environ.get("UC_SCHEMA", "llm_cache_app")

        # Create schema + table if needed, then insert
        stmts = [
            f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}",
            f"""CREATE TABLE IF NOT EXISTS {catalog}.{schema}.sales_log (
                sale_id STRING, customer_name STRING, product_id INT,
                quantity INT, sale_message STRING, created_at TIMESTAMP
            )""",
            f"""INSERT INTO {catalog}.{schema}.sales_log VALUES (
                '{sale_msg.split("#")[1].split(" ")[0] if "#" in sale_msg else "?"}',
                '{customer_name.replace("'", "''")}',
                {product_id}, {quantity},
                '{sale_msg.replace("'", "''")}',
                current_timestamp()
            )""",
        ]
        for sql in stmts:
            resp = requests.post(
                f"{host}/api/2.0/sql/statements",
                headers=headers,
                json={"statement": sql, "warehouse_id": os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID", "e5cdcc9f6056c833"), "wait_timeout": "30s"},
            )
            if not resp.ok:
                print(f"[agent] UC log SQL error: {resp.text[:200]}", flush=True)
                break
        else:
            print(f"[agent] Sale logged to UC: {catalog}.{schema}.sales_log", flush=True)
    except Exception as e:
        print(f"[agent] UC log failed (non-fatal): {e}", flush=True)


def _execute_tool(name, args, conn):
    if name == "create_sale":
        _, msg = db.create_sale(conn, args["product_id"], args["customer_name"], args.get("quantity", 1))
        if msg and "#" in msg:
            _log_sale_to_uc(args["product_id"], args["customer_name"], args.get("quantity", 1), msg)
        return msg
    elif name == "list_products":
        products = db.get_products(conn)
        cat = args.get("category", "").lower()
        if cat:
            products = [p for p in products if cat in p["category"].lower()]
        lines = []
        for p in products:
            lines.append(f"ID {p['id']}: {p['name']} | {p['category']} | R$ {float(p['price']):,.2f} | Estoque: {p['stock']} | {p['description']}")
        return "\n".join(lines) if lines else "Nenhum produto encontrado."
    elif name == "list_sales":
        sales = db.get_sales(conn)
        if not sales:
            return "Nenhuma venda registrada."
        lines = []
        for s in sales[:10]:
            lines.append(f"#{s['id']} | {s['customer_name']} | {s['product_name']} | {s['quantity']}x R$ {float(s['unit_price']):,.2f} = R$ {float(s['total']):,.2f}")
        return "\n".join(lines)
    return "Funcao nao reconhecida."


def _get_products_info(conn):
    products = db.get_products(conn)
    lines = []
    for p in products:
        lines.append(f"ID {p['id']}: {p['name']} ({p['category']}) - R$ {float(p['price']):,.2f} | Estoque: {p['stock']}")
    return "\n".join(lines)


def _get_guardrails_text(conn):
    guardrails = db.get_guardrails(conn)
    active = [g["rule_text"] for g in guardrails if g["enabled"]]
    if not active:
        return ""
    rules = "\n".join(f"- {r}" for r in active)
    return f"\nGUARDRAILS (regras obrigatorias):\n{rules}"


def call_llm(messages, conn, session_id=None):
    """Call LLM and handle tool calls. Returns (response_text, time_ms, tokens)."""
    client = _get_client()
    products_info = _get_products_info(conn)
    guardrails = _get_guardrails_text(conn)
    system_msg = SYSTEM_PROMPT.format(products_info=products_info, guardrails=guardrails)
    clean_msgs = []
    for m in messages:
        c = m.get("content")
        if c:
            clean_msgs.append({"role": m["role"], "content": c})
    full = [{"role": "system", "content": system_msg}] + clean_msgs

    start = time.time()
    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=full, tools=TOOLS, max_tokens=2048,
    )
    msg = response.choices[0].message
    tokens = response.usage.total_tokens if response.usage else 0

    while msg.tool_calls:
        tool_calls_list = []
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                tool_calls_list.append(tc)
                func = tc.get("function", {})
                tc_id, func_name, func_args = tc.get("id", ""), func.get("name", ""), func.get("arguments", "{}")
            else:
                tool_calls_list.append({
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                })
                tc_id, func_name, func_args = tc.id, tc.function.name, tc.function.arguments
        full.append({"role": "assistant", "tool_calls": tool_calls_list})
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                tc_id, func_name, func_args = tc.get("id", ""), func.get("name", ""), func.get("arguments", "{}")
            else:
                tc_id, func_name, func_args = tc.id, tc.function.name, tc.function.arguments
            args = json.loads(func_args)
            result = _execute_tool(func_name, args, conn)
            full.append({"role": "tool", "tool_call_id": tc_id, "content": result})

        response = client.chat.completions.create(
            model="databricks-claude-sonnet-4",
            messages=full, tools=TOOLS, max_tokens=2048,
        )
        msg = response.choices[0].message
        tokens += response.usage.total_tokens if response.usage else 0

    elapsed_ms = int((time.time() - start) * 1000)
    response_text = msg.content or ""

    # Tag the autolog trace with session and user info
    if _mlflow_enabled:
        try:
            tags = {}
            if session_id:
                tags["mlflow.trace.session"] = session_id
            if _mlflow_owner:
                tags["mlflow.user"] = _mlflow_owner
            if tags:
                mlflow.update_current_trace(tags=tags)
        except Exception as te:
            print(f"[agent] Trace tag error: {te}", flush=True)

    # Run guardrail eval synchronously (saves to DB + MLflow)
    if response_text:
        _run_guardrail_eval(messages[-1]["content"], response_text, conn)

    return response_text, elapsed_ms, tokens


def _run_guardrail_eval(user_message, agent_response, conn):
    """Evaluate response against guardrails and log to MLflow + Lakebase."""
    import traceback

    active_rules = db.get_guardrails(conn)
    enabled = [g for g in active_rules if g["enabled"]]
    if not enabled:
        print("[agent] No active guardrail rules, skipping eval", flush=True)
        return

    rule_texts = [g["rule_text"] for g in enabled]
    rule_names = [g.get("name") or f"regra_{i+1}" for i, g in enumerate(enabled)]

    print(f"[agent] Running guardrail eval with {len(rule_texts)} rules", flush=True)

    # Step 1: Call judge LLM directly (no MLflow dependency)
    judge_result = None
    try:
        import guardrail_eval
        judge_result = guardrail_eval._call_judge(user_message, agent_response, rule_texts)
        print(f"[agent] Judge done in {judge_result.get('eval_time_ms', 0)}ms", flush=True)
    except Exception as e:
        print(f"[agent] Judge call failed: {e}", flush=True)
        traceback.print_exc()
        return

    # Step 2: Save to DB FIRST (using fresh connection)
    if judge_result:
        try:
            fresh_conn = db.get_connection()
            guardrail_eval.save_evaluation_to_db(fresh_conn, user_message, agent_response, judge_result)
            fresh_conn.close()
            print(f"[agent] Eval saved to DB: pass={judge_result.get('overall_pass')}", flush=True)
        except Exception as e:
            print(f"[agent] DB save failed: {e}", flush=True)
            traceback.print_exc()

    # Step 3: Try MLflow logging (optional, can fail)
    if judge_result:
        try:
            guardrail_eval.evaluate_and_log(
                user_message, agent_response, rule_texts, rule_names,
                _precomputed_result=judge_result,
            )
        except Exception as e:
            print(f"[agent] MLflow log failed (non-fatal): {e}", flush=True)


def check_cache(messages, conn):
    """Check cache without calling LLM. Returns (response, time_ms, hit, similarity, matched_query) or None."""
    user_msg = messages[-1]["content"]

    if not db.is_cacheable_query(user_msg):
        return None

    cached_response, cached_time, similarity, matched_query = db.cache_get_exact(conn, user_msg)
    if cached_response:
        return cached_response, 1, True, similarity, matched_query

    try:
        query_embedding = get_embedding(user_msg)
        cached_response, cached_time, similarity, matched_query = db.cache_get_semantic(conn, user_msg, query_embedding)
        if cached_response:
            return cached_response, 1, True, similarity, matched_query
    except Exception:
        pass

    return None


def store_in_cache(conn, query_text, response_text, model="databricks-claude-sonnet-4", tokens=0, time_ms=0):
    """Store a response in cache (with embedding)."""
    if not db.is_cacheable_query(query_text):
        return
    try:
        embedding = get_embedding(query_text)
        db.cache_set(conn, query_text, response_text, embedding, model, tokens, time_ms)
    except Exception:
        db.cache_set(conn, query_text, response_text, None, model, tokens, time_ms)
