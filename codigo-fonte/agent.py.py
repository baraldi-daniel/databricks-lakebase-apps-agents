# Databricks notebook source
import json
import os
import time
from openai import OpenAI
import db

SYSTEM_PROMPT = """Voce e um agente de vendas de uma loja de tecnologia e eletronicos. Voce ajuda clientes a:
1. Consultar produtos disponiveis (nome, preco, estoque, categoria)
2. Registrar vendas (precisa do nome do cliente, produto, quantidade E email do cliente)
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
- IMPORTANTE: Para registrar uma venda, voce DEVE coletar o email do cliente antes de chamar create_sale. Pergunte o email se o cliente nao informou.
- Apos registrar a venda, use a funcao send_confirmation_email para enviar o email de confirmacao ao cliente.

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
    {
        "type": "function",
        "function": {
            "name": "send_confirmation_email",
            "description": "Envia email de confirmacao de compra ao cliente. Use APOS registrar a venda com create_sale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_email": {"type": "string", "description": "Email do cliente"},
                    "customer_name": {"type": "string", "description": "Nome do cliente"},
                    "product_name": {"type": "string", "description": "Nome do produto comprado"},
                    "quantity": {"type": "integer", "description": "Quantidade comprada"},
                    "unit_price": {"type": "number", "description": "Preco unitario"},
                    "total": {"type": "number", "description": "Valor total da compra"},
                },
                "required": ["customer_email", "customer_name", "product_name", "quantity", "unit_price", "total"],
            },
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


def _execute_tool(name, args, conn, email_callback=None):
    if name == "create_sale":
        _, msg = db.create_sale(conn, args["product_id"], args["customer_name"], args.get("quantity", 1))
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
    elif name == "send_confirmation_email":
        if email_callback:
            return email_callback(args)
        return "Email de confirmacao enviado com sucesso."
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


def call_llm(messages, conn, email_callback=None):
    """Call LLM and handle tool calls. Returns (response_text, time_ms, tokens)."""
    client = _get_client()
    products_info = _get_products_info(conn)
    guardrails = _get_guardrails_text(conn)
    system_msg = SYSTEM_PROMPT.format(products_info=products_info, guardrails=guardrails)
    # Filter messages: skip any with None/empty content (e.g. leftover tool call messages)
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
        # Convert tool call message to dict for Databricks API compatibility
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
            result = _execute_tool(func_name, args, conn, email_callback)
            full.append({"role": "tool", "tool_call_id": tc_id, "content": result})

        response = client.chat.completions.create(
            model="databricks-claude-sonnet-4",
            messages=full, tools=TOOLS, max_tokens=2048,
        )
        msg = response.choices[0].message
        tokens += response.usage.total_tokens if response.usage else 0

    elapsed_ms = int((time.time() - start) * 1000)
    response_text = msg.content or ""

    # Run guardrail evaluation asynchronously
    if response_text:
        _run_guardrail_eval(messages[-1]["content"], response_text, conn)

    return response_text, elapsed_ms, tokens


def _run_guardrail_eval(user_message, agent_response, conn):
    """Evaluate response against guardrails and log to MLflow + Lakebase."""
    import threading
    import traceback

    active_rules = db.get_guardrails(conn)
    enabled = [g for g in active_rules if g["enabled"]]
    if not enabled:
        print("[agent] No active guardrail rules, skipping eval", flush=True)
        return

    rule_texts = [g["rule_text"] for g in enabled]
    rule_names = [g.get("name") or f"regra_{i+1}" for i, g in enumerate(enabled)]

    print(f"[agent] Starting guardrail eval thread with {len(rule_texts)} rules", flush=True)

    def _eval():
        try:
            import guardrail_eval
            eval_result = guardrail_eval.evaluate_and_log(user_message, agent_response, rule_texts, rule_names)
            if eval_result:
                eval_conn = db.get_connection()
                guardrail_eval.save_evaluation_to_db(eval_conn, user_message, agent_response, eval_result)
                eval_conn.close()
                print(f"[agent] Guardrail eval saved to DB: pass={eval_result.get('overall_pass')}", flush=True)
            else:
                print("[agent] Guardrail eval returned None", flush=True)
        except Exception as e:
            print(f"[agent] Guardrail eval error: {e}", flush=True)
            traceback.print_exc()

    threading.Thread(target=_eval, daemon=True).start()


def check_cache(messages, conn):
    """Check cache without calling LLM. Returns (response, time_ms, hit, similarity, matched_query) or None."""
    user_msg = messages[-1]["content"]

    # Smart cache: skip for specific/personal queries
    if not db.is_cacheable_query(user_msg):
        return None

    # Tier 1: Exact hash match
    cached_response, cached_time, similarity, matched_query = db.cache_get_exact(conn, user_msg)
    if cached_response:
        return cached_response, 1, True, similarity, matched_query

    # Tier 2: Semantic similarity (requires embedding API)
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