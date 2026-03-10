# Databricks notebook source
import streamlit as st
import pandas as pd
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import db
import agent
import gmail_auth
import cache_graph

st.set_page_config(page_title="LLM Semantic Cache Demo", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .cache-hit {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        color: white; padding: 5px 14px; border-radius: 20px;
        font-size: 0.8em; font-weight: 600; display: inline-block;
    }
    .cache-miss {
        background: linear-gradient(135deg, #ff6d00 0%, #ff9100 100%);
        color: white; padding: 5px 14px; border-radius: 20px;
        font-size: 0.8em; font-weight: 600; display: inline-block;
    }
    .cache-exact {
        background: linear-gradient(135deg, #2196f3 0%, #42a5f5 100%);
        color: white; padding: 5px 14px; border-radius: 20px;
        font-size: 0.8em; font-weight: 600; display: inline-block;
    }
    .cache-skip {
        background: linear-gradient(135deg, #9e9e9e 0%, #bdbdbd 100%);
        color: white; padding: 5px 14px; border-radius: 20px;
        font-size: 0.8em; font-weight: 600; display: inline-block;
    }
    .time-badge {
        background: #1e1e2e; color: #cdd6f4; padding: 4px 10px;
        border-radius: 12px; font-size: 0.78em; font-family: monospace;
        display: inline-block; margin-left: 6px;
    }
    .speed-gain {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
        color: white; padding: 5px 14px; border-radius: 20px;
        font-size: 0.82em; font-weight: 700; display: inline-block; margin-left: 6px;
    }
    .sim-badge {
        background: #1e3a5f; color: #7dd3fc; padding: 4px 10px;
        border-radius: 12px; font-size: 0.75em; font-family: monospace;
        display: inline-block; margin-left: 6px;
    }
    .matched-query {
        color: #6c7086; font-size: 0.75em; font-style: italic; margin-top: 4px;
    }
    .waiting-badge {
        background: #313244; color: #a6adc8; padding: 5px 14px;
        border-radius: 20px; font-size: 0.8em; display: inline-block;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
for key in ["msgs_nc", "msgs_c", "chat_nc", "chat_c", "times_nc", "times_c"]:
    if key not in st.session_state:
        st.session_state[key] = []


def get_conn():
    return db.get_connection()


def get_email_callback():
    """Create email callback if Gmail is connected."""
    gmail_email = st.session_state.get("gmail_email", "")
    if not gmail_email:
        return None
    try:
        conn = get_conn()
        token = gmail_auth.get_valid_token(conn, gmail_email)
        conn.close()
        if not token:
            return None
    except Exception:
        return None

    def send_email(args):
        try:
            subject = f"Confirmacao de Compra - {args['product_name']}"
            body = gmail_auth.build_sale_email(
                args["product_name"], args["quantity"],
                args["unit_price"], args["total"], args["customer_name"]
            )
            gmail_auth.send_email(token, args["customer_email"], subject, body)
            return f"Email de confirmacao enviado para {args['customer_email']}"
        except Exception as e:
            return f"Erro ao enviar email: {e}"
    return send_email


def render_nc_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content") or "")
        if msg["role"] == "assistant" and "time_ms" in msg:
            if msg.get("waiting"):
                st.markdown('<span class="waiting-badge">Aguardando LLM...</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="time-badge">{msg["time_ms"]:,}ms</span>', unsafe_allow_html=True)


def render_c_message(msg, idx):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "time_ms" in msg:
            hit = msg.get("cache_hit", False)
            sim = msg.get("similarity", 0)
            skipped = msg.get("cache_skipped", False)
            if skipped:
                badge = '<span class="cache-skip">CACHE SKIP</span>'
            elif hit and sim >= 1.0:
                badge = '<span class="cache-exact">EXACT MATCH</span>'
            elif hit:
                badge = '<span class="cache-hit">CACHE HIT</span>'
            else:
                badge = '<span class="cache-miss">CACHE MISS</span>'
            time_b = f'<span class="time-badge">{msg["time_ms"]:,}ms</span>'
            sim_b = f'<span class="sim-badge">similaridade: {sim:.3f}</span>' if hit and 0 < sim < 1.0 else ""
            speed = ""
            if idx < len(st.session_state.times_nc) and msg["time_ms"] > 0:
                nc_time = st.session_state.times_nc[idx]
                if nc_time > 0:
                    factor = nc_time / msg["time_ms"]
                    if factor > 1.5:
                        speed = f'<span class="speed-gain">{factor:,.0f}x mais rapido</span>'
            st.markdown(f'{badge} {time_b} {sim_b} {speed}', unsafe_allow_html=True)
            if hit and msg.get("matched_query"):
                st.markdown(f'<p class="matched-query">Pergunta cacheada: "{msg["matched_query"]}"</p>', unsafe_allow_html=True)
            if msg.get("cache_analysis"):
                st.markdown(f'<p class="matched-query">Analise LLM: {msg["cache_analysis"]}</p>', unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["Comparacao Side-by-Side", "Conversas & Cache", "Configuracoes"])

# ========== TAB 1 ==========
with tab1:
    st.markdown("## LLM Semantic Cache — Agente de Vendas")
    st.caption("Cache semantico com dois niveis: hash exato (instantaneo) e embeddings + cosine similarity (threshold >= 0.92). Queries especificas (com nomes, emails) pulam o cache automaticamente.")

    try:
        conn = get_conn()
        stats = db.get_cache_stats(conn)
        conn.close()
    except Exception:
        stats = {"entries": 0, "hits": 0, "avg_ms": 0}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cache Entries", stats["entries"])
    c2.metric("Total Cache Hits", stats["hits"])
    avg_nc = int(sum(st.session_state.times_nc) / len(st.session_state.times_nc)) if st.session_state.times_nc else 0
    avg_c = int(sum(st.session_state.times_c) / len(st.session_state.times_c)) if st.session_state.times_c else 0
    c3.metric("Media sem Cache", f"{avg_nc:,}ms")
    c4.metric("Media com Cache", f"{avg_c:,}ms")

    st.divider()

    col_nc, col_c = st.columns(2)

    with col_nc:
        st.markdown("### Sem Cache (LLM direto)")
        cont_nc = st.container(height=440)
        with cont_nc:
            for msg in st.session_state.chat_nc:
                render_nc_message(msg)

    with col_c:
        st.markdown("### Com Semantic Cache (Lakebase)")
        cont_c = st.container(height=440)
        with cont_c:
            assistant_idx = 0
            for msg in st.session_state.chat_c:
                if msg["role"] == "assistant":
                    render_c_message(msg, assistant_idx)
                    assistant_idx += 1
                else:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

    # ---- Check if background LLM call finished ----
    if "llm_future" in st.session_state:
        future = st.session_state.llm_future
        if future.done():
            if st.session_state.chat_nc and st.session_state.chat_nc[-1].get("waiting"):
                st.session_state.chat_nc.pop()
            try:
                resp, time_ms, tokens = future.result()
                # Non-cached side
                st.session_state.chat_nc.append({"role": "assistant", "content": resp or "", "time_ms": time_ms})
                st.session_state.msgs_nc.append({"role": "assistant", "content": resp or ""})
                st.session_state.times_nc.append(time_ms)

                # Cached side on miss — reuse same LLM result
                if st.session_state.get("cache_miss_pending"):
                    if st.session_state.chat_c and st.session_state.chat_c[-1].get("waiting"):
                        st.session_state.chat_c.pop()
                    analysis = st.session_state.get("cache_analysis", "")
                    is_cacheable = st.session_state.get("cache_is_cacheable", False)
                    st.session_state.chat_c.append({
                        "role": "assistant", "content": resp or "", "time_ms": time_ms,
                        "cache_hit": False, "similarity": 0,
                        "cache_skipped": not is_cacheable,
                        "cache_analysis": analysis,
                    })
                    st.session_state.msgs_c.append({"role": "assistant", "content": resp or ""})
                    st.session_state.times_c.append(time_ms)
                    # Store in cache if the LLM analyzer said it's cacheable
                    if is_cacheable and resp and st.session_state.get("last_query"):
                        try:
                            conn_store = get_conn()
                            agent.store_in_cache(conn_store, st.session_state["last_query"], resp,
                                                  tokens=tokens, time_ms=time_ms)
                            conn_store.close()
                        except Exception:
                            pass
                    del st.session_state["cache_miss_pending"]
                    if "cache_analysis" in st.session_state:
                        del st.session_state["cache_analysis"]
                    if "cache_is_cacheable" in st.session_state:
                        del st.session_state["cache_is_cacheable"]
            except Exception as e:
                if st.session_state.chat_nc and st.session_state.chat_nc[-1].get("waiting"):
                    st.session_state.chat_nc.pop()
                st.session_state.chat_nc.append({"role": "assistant", "content": f"Erro: {e}", "time_ms": 0})
                st.session_state.times_nc.append(0)
                if st.session_state.get("cache_miss_pending"):
                    if st.session_state.chat_c and st.session_state.chat_c[-1].get("waiting"):
                        st.session_state.chat_c.pop()
                    st.session_state.chat_c.append({
                        "role": "assistant", "content": f"Erro: {e}", "time_ms": 0,
                        "cache_hit": False, "similarity": 0
                    })
                    st.session_state.times_c.append(0)
                    del st.session_state["cache_miss_pending"]
            del st.session_state["llm_future"]
            if "last_query" in st.session_state:
                del st.session_state["last_query"]
            st.rerun()
        else:
            time.sleep(0.5)
            st.rerun()

    # ---- On new prompt ----
    prompt = st.chat_input("Pergunte ao agente de vendas... (ex: Quais notebooks voces tem?)")

    if prompt:
        st.session_state.chat_nc.append({"role": "user", "content": prompt})
        st.session_state.chat_c.append({"role": "user", "content": prompt})
        st.session_state.msgs_nc.append({"role": "user", "content": prompt})
        st.session_state.msgs_c.append({"role": "user", "content": prompt})
        st.session_state["last_query"] = prompt

        # Save user messages to DB
        try:
            conn_save = get_conn()
            db.save_message(conn_save, st.session_state.session_id + "-nc", "user", prompt, cached=False)
            db.save_message(conn_save, st.session_state.session_id + "-c", "user", prompt, cached=True)
            conn_save.close()
        except Exception:
            pass

        # Run LangGraph cache analysis + lookup (synchronous — fast)
        email_cb = get_email_callback()
        session_id = st.session_state.session_id

        cache_resp, cache_time, is_hit, similarity, matched_query, is_cacheable, analysis = (
            None, 0, False, 0.0, None, False, ""
        )
        try:
            cache_resp, cache_time, is_hit, similarity, matched_query, is_cacheable, analysis = (
                cache_graph.run_cache_check(list(st.session_state.msgs_c))
            )
        except Exception:
            pass

        # Start single LLM call in background (used by both sides on cache miss)
        def run_llm_direct(msgs, sid):
            conn_llm = db.get_connection()
            result = agent.call_llm(msgs, conn_llm, email_cb)
            db.save_message(conn_llm, sid + "-nc", "assistant",
                            result[0] or "", cached=False, response_time_ms=result[1])
            conn_llm.close()
            return result

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_llm_direct, list(st.session_state.msgs_nc), session_id)
        st.session_state.llm_future = future
        st.session_state["cache_is_cacheable"] = is_cacheable
        st.session_state["cache_analysis"] = analysis

        if is_hit:
            # Cache HIT — show immediately on cached side
            st.session_state.chat_c.append({
                "role": "assistant", "content": cache_resp, "time_ms": cache_time,
                "cache_hit": True, "similarity": similarity,
                "matched_query": matched_query, "cache_analysis": analysis,
            })
            st.session_state.msgs_c.append({"role": "assistant", "content": cache_resp})
            st.session_state.times_c.append(cache_time)
            try:
                conn_hit = get_conn()
                db.save_message(conn_hit, session_id + "-c", "assistant",
                                cache_resp, cached=True, response_time_ms=1,
                                cache_hit=True, similarity=similarity)
                conn_hit.close()
            except Exception:
                pass
            st.session_state["cache_miss_pending"] = False
        else:
            # Cache MISS or SKIP — cached side waits for the same LLM result
            st.session_state["cache_miss_pending"] = True
            st.session_state.chat_c.append({
                "role": "assistant", "content": "Aguardando LLM...", "time_ms": 0, "waiting": True
            })

        # Non-cached side always waits
        st.session_state.chat_nc.append({
            "role": "assistant", "content": "Processando...", "time_ms": 0, "waiting": True
        })

        st.rerun()

# ========== TAB 2 ==========
with tab2:
    st.markdown("## Conversas com Cache & Cache Entries")

    try:
        conn = get_conn()
        cache_entries = db.get_cache_entries(conn)
        conversations = db.get_conversations(conn)
        conn.close()
    except Exception as e:
        st.error(f"Erro: {e}")
        cache_entries = []
        conversations = []

    st.subheader("Entradas do Semantic Cache")
    if cache_entries:
        df_cache = pd.DataFrame([{
            "Pergunta": e["query_text"][:100],
            "Resposta": e["response_text"][:120] + "...",
            "Hits": e["hit_count"],
            "Tempo Original (ms)": e["response_time_ms"],
            "Criado em": e["created_at"],
            "Ultimo Hit": e["last_hit_at"],
        } for e in cache_entries])
        st.dataframe(df_cache, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma entrada no cache ainda. Faca perguntas na aba anterior!")

    st.divider()

    st.subheader("Avaliacoes de Guardrails (LLM-as-a-Judge)")
    try:
        conn_eval = get_conn()
        import guardrail_eval
        evaluations = guardrail_eval.get_recent_evaluations(conn_eval, limit=20)
        conn_eval.close()
    except Exception:
        evaluations = []

    if evaluations:
        for ev in evaluations:
            status = "PASS" if ev["overall_pass"] else "FAIL"
            color = "cache-hit" if ev["overall_pass"] else "cache-miss"
            with st.expander(
                f"[{status}] {ev['user_message'][:80]}... | {ev['created_at'].strftime('%d/%m %H:%M') if ev.get('created_at') else ''}",
                expanded=False
            ):
                st.markdown(f'<span class="{color}">{status}</span> <span class="time-badge">Judge: {ev["eval_time_ms"]}ms</span>', unsafe_allow_html=True)
                st.caption(f"Pass rate: {ev['pass_rate']:.0%}")
                st.markdown(f"**Pergunta:** {ev['user_message'][:200]}")
                st.markdown(f"**Resposta:** {ev['agent_response'][:300]}...")
                st.markdown(f"**Resumo:** {ev['summary']}")
                if ev.get("evaluations_json"):
                    import json as _json
                    try:
                        rule_evals = _json.loads(ev["evaluations_json"])
                        for re_item in rule_evals:
                            icon = "V" if re_item["verdict"] == "pass" else ("X" if re_item["verdict"] == "fail" else "-")
                            st.markdown(f"- [{icon}] **{re_item['verdict'].upper()}** - {re_item['rule']}: {re_item.get('reason', '')}")
                    except Exception:
                        pass
        st.caption("Avaliacoes tambem registradas no MLflow: /Users/<YOUR_EMAIL>/guardrail-evaluations")
    else:
        st.info("Nenhuma avaliacao de guardrail ainda. Adicione regras na aba Configuracoes e converse com o agente.")

    st.divider()

    st.subheader("Historico de Conversas (com Cache)")
    if conversations:
        sessions = {}
        for c in conversations:
            sid = c["session_id"]
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(c)

        for sid, msgs in sessions.items():
            with st.expander(f"Sessao {sid} - {len(msgs)} mensagens", expanded=len(sessions) == 1):
                for m in reversed(msgs):
                    icon = "U" if m["role"] == "user" else "A"
                    extra = ""
                    if m["role"] == "assistant":
                        if m["cache_hit"]:
                            sim = m.get("similarity", 0)
                            extra = f' <span class="cache-hit">CACHE HIT</span> <span class="sim-badge">sim: {sim:.3f}</span>'
                        elif m["response_time_ms"] > 0:
                            extra = f' <span class="cache-miss">CACHE MISS</span>'
                        extra += f' <span class="time-badge">{m["response_time_ms"]:,}ms</span>'
                    content = m.get('content') or ''
                    st.markdown(f"**[{icon}] {m['role'].title()}:** {content[:400]}{extra}", unsafe_allow_html=True)
    else:
        st.info("Nenhuma conversa registrada ainda.")

# ========== TAB 3 ==========
with tab3:
    st.markdown("## Configuracoes")

    col_left, col_right = st.columns(2)

    # ---- Guardrails ----
    with col_left:
        st.subheader("Guardrails do Agente")
        st.caption("Configure regras em linguagem natural. O agente seguira essas regras em todas as interacoes.")

        try:
            conn_g = get_conn()
            guardrails = db.get_guardrails(conn_g)
        except Exception:
            conn_g = None
            guardrails = []

        if conn_g:
            for g in guardrails:
                gid = g["id"]
                with st.container(border=True):
                    c_name, c_toggle, c_del = st.columns([0.6, 0.2, 0.2])
                    with c_name:
                        new_name_val = st.text_input(
                            "Nome", value=g.get("name") or "", key=f"name_{gid}",
                            placeholder="Nome da regra")
                    with c_toggle:
                        st.write("")
                        new_enabled = st.toggle("Ativo", value=g["enabled"], key=f"toggle_{gid}")
                        if new_enabled != g["enabled"]:
                            db.toggle_guardrail(conn_g, gid, new_enabled)
                            st.rerun()
                    with c_del:
                        st.write("")
                        st.write("")
                        if st.button("Remover", key=f"del_{gid}", use_container_width=True):
                            gname = g.get("name", "")
                            if gname:
                                try:
                                    import guardrail_eval
                                    guardrail_eval.unregister_judge(gname.replace(" ", "_").replace("/", "_"))
                                except Exception:
                                    pass
                            db.delete_guardrail(conn_g, gid)
                            st.rerun()
                    new_rule_val = st.text_area(
                        "Regra", value=g["rule_text"], key=f"rule_{gid}",
                        height=68)
                    if st.button("Salvar", key=f"save_{gid}", type="primary", use_container_width=True):
                        db.update_guardrail(conn_g, gid, name=new_name_val, rule_text=new_rule_val)
                        st.success("Salvo!")
                        st.rerun()

            st.divider()
            st.markdown("**Nova regra**")
            col_name, col_rule = st.columns([0.3, 0.7])
            with col_name:
                new_name = st.text_input("Nome",
                                          placeholder="Ex: idioma, tom, dados",
                                          key="new_guardrail_name")
            with col_rule:
                new_rule = st.text_area("Regra (linguagem natural)",
                                         placeholder="Ex: Nao fale sobre politica ou religiao",
                                         height=80, key="new_guardrail_rule")
            if st.button("Adicionar Regra"):
                if new_rule.strip():
                    rule_name = new_name.strip() or new_rule.strip()[:30]
                    db.add_guardrail(conn_g, new_rule.strip(), name=rule_name)
                    try:
                        import guardrail_eval
                        guardrail_eval.register_judge(rule_name.replace(" ", "_").replace("/", "_"), new_rule.strip())
                    except Exception:
                        pass
                    st.rerun()

            conn_g.close()

    # ---- Gmail ----
    with col_right:
        st.subheader("Gmail - Emails de Confirmacao")
        st.caption("Conecte seu Gmail para enviar emails de confirmacao de compra aos clientes.")

        try:
            conn_gmail = get_conn()
        except Exception:
            conn_gmail = None

        if conn_gmail:
            query_params = st.query_params
            auth_code = query_params.get("code")

            if auth_code and "gmail_auth_done" not in st.session_state:
                try:
                    import os
                    app_url = os.environ.get("DATABRICKS_APP_URL", "http://localhost:8000")
                    redirect_uri = app_url.rstrip("/")
                    tokens = gmail_auth.exchange_code(auth_code, redirect_uri)
                    access_token = tokens["access_token"]
                    refresh_token = tokens.get("refresh_token", "")
                    expires_in = tokens.get("expires_in", 3600)
                    token_expiry = datetime.now() + timedelta(seconds=expires_in)
                    user_email = gmail_auth.get_user_email(access_token)
                    ttl = st.session_state.get("gmail_ttl", 720)
                    db.save_gmail_token(conn_gmail, user_email, access_token, refresh_token, token_expiry, ttl)
                    st.session_state.gmail_email = user_email
                    st.session_state.gmail_auth_done = True
                    st.query_params.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro na autenticacao: {e}")

            gmail_email = st.session_state.get("gmail_email", "")
            if gmail_email:
                token_data = db.get_gmail_token(conn_gmail, gmail_email)
                if token_data:
                    valid_token = gmail_auth.get_valid_token(conn_gmail, gmail_email)
                    if valid_token:
                        st.success(f"Conectado como: {gmail_email}")
                        ttl_hours = token_data["ttl_hours"] or 720
                        created = token_data["created_at"]
                        if created:
                            expires_at = created + timedelta(hours=ttl_hours)
                            st.caption(f"Sessao expira em: {expires_at.strftime('%d/%m/%Y %H:%M')}")
                        if st.button("Desconectar Gmail"):
                            db.delete_gmail_token(conn_gmail, gmail_email)
                            del st.session_state["gmail_email"]
                            if "gmail_auth_done" in st.session_state:
                                del st.session_state["gmail_auth_done"]
                            st.rerun()
                    else:
                        st.warning("Token expirado. Reconecte abaixo.")
                        gmail_email = ""

            if not gmail_email:
                ttl = st.number_input("TTL da sessao (horas)", min_value=1, max_value=8760, value=720, step=24,
                                      help="Tempo em horas ate a sessao expirar automaticamente.")
                st.session_state["gmail_ttl"] = ttl
                import os
                app_url = os.environ.get("DATABRICKS_APP_URL", "http://localhost:8000")
                redirect_uri = app_url.rstrip("/")
                auth_url = gmail_auth.get_auth_url(redirect_uri)
                st.markdown(f"[Conectar Gmail]({auth_url})")

            conn_gmail.close()
        else:
            st.error("Sem conexao com o banco de dados.")