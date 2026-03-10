# Databricks notebook source
"""LangGraph-based cache routing with LLM analysis for cacheability decisions."""
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

    # Fast pre-filter: obvious non-cacheable patterns
    if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', user_query):
        return {"is_cacheable": False, "cache_analysis": "Contem email"}
    if re.search(r'#\d+|venda\s+\d+|pedido\s+\d+', user_query, re.IGNORECASE):
        return {"is_cacheable": False, "cache_analysis": "Referencia pedido/venda"}

    # Build conversation context
    msgs = state["messages"]
    conv_lines = []
    for m in msgs[:-1]:
        c = m.get("content", "")
        if c:
            role = "Usuario" if m["role"] == "user" else "Agente"
            conv_lines.append(f"{role}: {c[:200]}")
    conversation = "\n".join(conv_lines[-10:]) if conv_lines else "(primeira mensagem)"

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
        # Tier 1: Exact hash
        cached_resp, cached_time, similarity, matched_query = db.cache_get_exact(conn, state["user_query"])
        if cached_resp:
            return {"cache_result": {
                "response": cached_resp, "time_ms": 1, "hit": True,
                "similarity": similarity, "matched_query": matched_query
            }}

        # Tier 2: Semantic similarity
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