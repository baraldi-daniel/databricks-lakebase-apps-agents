# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4: Register Guardrails as Judges in MLflow
# MAGIC
# MAGIC This notebook registers existing guardrail rules as **LLM Judges**
# MAGIC in the MLflow experiment, so they appear in the **Judges** tab and evaluate traces automatically.
# MAGIC
# MAGIC **When to use**: Whenever you create guardrails directly in the database (without using the app UI).
# MAGIC The app UI already registers judges automatically when creating a new rule.

# COMMAND ----------

# ============================================================
# FILL IN ONLY THESE 2 VALUES (same as notebook 01)
# ============================================================
LAKEBASE_CONNECTION = "<PASTE_CONNECTION_STRING_OR_HOST>"  # postgresql://user@ep-xxx... or just the host
LAKEBASE_PASSWORD = "<SAME_PASSWORD_FROM_NOTEBOOK_01>"

# --- Auto-detected ---
from databricks.sdk import WorkspaceClient
from urllib.parse import urlparse
_w = WorkspaceClient()
USER_EMAIL = _w.current_user.me().user_name
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/guardrail-evaluations"

if LAKEBASE_CONNECTION.startswith("postgresql://") or LAKEBASE_CONNECTION.startswith("postgres://"):
    LAKEBASE_HOST = urlparse(LAKEBASE_CONNECTION).hostname
else:
    LAKEBASE_HOST = LAKEBASE_CONNECTION.strip()

print(f"User: {USER_EMAIL}")
print(f"Lakebase Host: {LAKEBASE_HOST}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 — Read guardrails from Lakebase

# COMMAND ----------

import psycopg

conn = psycopg.connect(
    host=LAKEBASE_HOST, port=5432, dbname="databricks_postgres",
    user="cache_app", password=LAKEBASE_PASSWORD, sslmode="require", autocommit=True,
)
cur = conn.cursor()
cur.execute("SELECT id, name, rule_text, enabled FROM guardrails WHERE enabled = TRUE ORDER BY id")
guardrails = cur.fetchall()
cur.close()
conn.close()

print(f"Found {len(guardrails)} active rules:")
for g in guardrails:
    print(f"  [{g[0]}] {g[1] or '(no name)'}: {g[2][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 — Register as Judges in MLflow

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import Guidelines, ScorerSamplingConfig

mlflow.set_experiment(EXPERIMENT_PATH)

for gid, name, rule_text, enabled in guardrails:
    judge_name = (name or f"rule_{gid}").replace(" ", "_").replace("/", "_")
    try:
        judge = Guidelines(name=judge_name, guidelines=rule_text)
        registered = judge.register(name=judge_name)
        registered.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
        print(f"Judge '{judge_name}' registered and activated!")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Judge '{judge_name}' already exists, skipping.")
        else:
            print(f"Error with '{judge_name}': {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 — Verify registered judges

# COMMAND ----------

from mlflow.genai.scorers import get_scorer

for gid, name, rule_text, enabled in guardrails:
    judge_name = (name or f"rule_{gid}").replace(" ", "_").replace("/", "_")
    try:
        s = get_scorer(name=judge_name)
        print(f"Judge '{judge_name}': OK")
    except Exception:
        print(f"Judge '{judge_name}': NOT FOUND")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 — Remove a judge (if needed)
# MAGIC
# MAGIC Uncomment and run to remove a specific judge.

# COMMAND ----------

# from mlflow.genai.scorers import delete_scorer
# delete_scorer(name="judge_name_here")
# print("Judge removed!")