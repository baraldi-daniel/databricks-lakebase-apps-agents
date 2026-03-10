# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Deploy the Application
# MAGIC
# MAGIC This notebook deploys the Databricks App using the SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Have run notebook `01_setup_lakebase`
# MAGIC 2. Have run notebook `02_source_code`
# MAGIC 3. Have the app files in `/Workspace/Users/<email>/llm-cache-app/`
# MAGIC
# MAGIC ## Manual steps (via terminal/CLI):
# MAGIC
# MAGIC ```bash
# MAGIC # 1. Authenticate with the workspace
# MAGIC databricks auth login --host <WORKSPACE_URL> --profile <PROFILE>
# MAGIC
# MAGIC # 2. Upload files (if not done yet)
# MAGIC databricks workspace import-dir ./llm-cache-app /Workspace/Users/<EMAIL>/llm-cache-app \
# MAGIC   --overwrite --profile <PROFILE>
# MAGIC
# MAGIC # 3. Create the app
# MAGIC databricks apps create <APP_NAME> \
# MAGIC   --description "Sales Agent with LLM Semantic Cache" \
# MAGIC   --profile <PROFILE>
# MAGIC
# MAGIC # 4. Note the returned URL and update DATABRICKS_APP_URL in app.yaml
# MAGIC
# MAGIC # 5. Re-upload the updated app.yaml
# MAGIC databricks workspace import-dir ./llm-cache-app /Workspace/Users/<EMAIL>/llm-cache-app \
# MAGIC   --overwrite --profile <PROFILE>
# MAGIC
# MAGIC # 6. Deploy
# MAGIC databricks apps deploy <APP_NAME> \
# MAGIC   --source-code-path /Workspace/Users/<EMAIL>/llm-cache-app \
# MAGIC   --profile <PROFILE>
# MAGIC
# MAGIC # 7. Add SQL Warehouse resource (via UI):
# MAGIC #    Compute > Apps > your-app > Settings > Add Resource > SQL Warehouse
# MAGIC
# MAGIC # 8. Verify
# MAGIC databricks apps get <APP_NAME> --profile <PROFILE>
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Or run via Python (in this notebook):

# COMMAND ----------

# ============================================================
# FILL IN ONLY THESE 3 VALUES (same as notebook 01)
# ============================================================
APP_NAME = "llm-caching"  # e.g. llm-caching, smart-cache, my-agent
LAKEBASE_CONNECTION = "<PASTE_CONNECTION_STRING>"  # postgresql://user@ep-xxx... or just the host
LAKEBASE_PASSWORD = "<YOUR_PASSWORD>"  # the password you chose in step 1

# --- Auto-detected ---
from databricks.sdk import WorkspaceClient
from urllib.parse import urlparse
_w = WorkspaceClient()
USER_EMAIL = _w.current_user.me().user_name
WORKSPACE_URL = _w.config.host

# Parse connection string or use as host directly
if LAKEBASE_CONNECTION.startswith("postgresql://") or LAKEBASE_CONNECTION.startswith("postgres://"):
    LAKEBASE_HOST = urlparse(LAKEBASE_CONNECTION).hostname
else:
    LAKEBASE_HOST = LAKEBASE_CONNECTION.strip()

print(f"User: {USER_EMAIL}")
print(f"Workspace: {WORKSPACE_URL}")
print(f"Lakebase Host: {LAKEBASE_HOST}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 — Update app.yaml with real values

# COMMAND ----------

APP_DIR = f"/Workspace/Users/{USER_EMAIL}/llm-cache-app"
yaml_path = f"{APP_DIR}/app.yaml"

# Write app.yaml with real values
with open(yaml_path, "w") as f:
    f.write(f"""command:
  - streamlit
  - run
  - app.py
  - --server.port
  - "8000"
  - --server.address
  - "0.0.0.0"

env:
  - name: DATABRICKS_HOST
    value: "{WORKSPACE_URL}"
  - name: LAKEBASE_HOST
    value: "{LAKEBASE_HOST}"
  - name: LAKEBASE_PG_USER
    value: "cache_app"
  - name: LAKEBASE_PG_PASSWORD
    value: "{LAKEBASE_PASSWORD}"
  - name: DATABRICKS_APP_URL
    value: "WILL_BE_FILLED_AFTER_CREATE"
  - name: APP_OWNER_EMAIL
    value: "{USER_EMAIL}"
  - name: DATABRICKS_SQL_WAREHOUSE_ID
    value: "e5cdcc9f6056c833"
  - name: UC_CATALOG
    value: "baraldiworkspace_catalog"
  - name: UC_SCHEMA
    value: "llm_cache_app"

resources:
  - name: sql-warehouse
    sql_warehouse:
      permission: CAN_USE
""")

print("app.yaml created with real values!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 — Create the app via SDK

# COMMAND ----------

w = _w  # reuse client from config cell

import json, requests

headers = w.config.authenticate()
if callable(headers):
    h = {}
    headers(h)
    headers = h

resp = requests.post(
    f"{WORKSPACE_URL}/api/2.0/apps",
    headers=headers,
    json={"name": APP_NAME, "description": "Sales Agent with LLM Semantic Cache"},
)
if resp.status_code == 409:
    # App already exists
    resp = requests.get(f"{WORKSPACE_URL}/api/2.0/apps/{APP_NAME}", headers=headers)
    app_data = resp.json()
    print(f"App already exists. URL: {app_data.get('url', '')}")
else:
    resp.raise_for_status()
    app_data = resp.json()
    print(f"App created: {APP_NAME}")
    print(f"URL: {app_data.get('url', 'pending...')}")

app_url = app_data.get("url", "")
sp_client_id = app_data.get("service_principal_client_id", "")

# Update DATABRICKS_APP_URL in yaml
if app_url:
    with open(yaml_path, "r") as f:
        content = f.read()
    content = content.replace("WILL_BE_FILLED_AFTER_CREATE", app_url)
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"app.yaml updated with URL: {app_url}")

# Create MLflow experiment and grant SP permissions (via REST API — no mlflow install needed)
exp_path = f"/Users/{USER_EMAIL}/guardrail-evaluations"
exp_resp = requests.post(
    f"{WORKSPACE_URL}/api/2.0/mlflow/experiments/create",
    headers=headers,
    json={"name": exp_path},
)
if exp_resp.ok:
    exp_id = exp_resp.json().get("experiment_id", "")
    print(f"MLflow experiment created: {exp_path} (id={exp_id})")
elif "RESOURCE_ALREADY_EXISTS" in exp_resp.text:
    # Get existing experiment ID
    get_resp = requests.get(
        f"{WORKSPACE_URL}/api/2.0/mlflow/experiments/get-by-name",
        headers=headers,
        params={"experiment_name": exp_path},
    )
    exp_id = get_resp.json().get("experiment", {}).get("experiment_id", "")
    print(f"MLflow experiment already exists: {exp_path} (id={exp_id})")
else:
    exp_id = ""
    print(f"Warning creating experiment: {exp_resp.text[:200]}")

if exp_id and sp_client_id:
    resp_perm = requests.patch(
        f"{WORKSPACE_URL}/api/2.0/permissions/experiments/{exp_id}",
        headers=headers,
        json={"access_control_list": [
            {"service_principal_name": sp_client_id, "permission_level": "CAN_MANAGE"}
        ]},
    )
    if resp_perm.ok:
        print(f"SP permissions granted on experiment!")
    else:
        print(f"Warning: could not set SP permissions: {resp_perm.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 — Deploy

# COMMAND ----------

source_path = f"/Workspace/Users/{USER_EMAIL}/llm-cache-app"
resp = requests.post(
    f"{WORKSPACE_URL}/api/2.0/apps/{APP_NAME}/deployments",
    headers=headers,
    json={"source_code_path": source_path},
)
if not resp.ok:
    if "active deployment in progress" in resp.text:
        print("Deploy already in progress — skipping to status check.")
    else:
        print(f"Error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
else:
    deploy_data = resp.json()
    print(f"Deploy started: {deploy_data.get('deployment_id', '')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 — Check status

# COMMAND ----------

import time

for i in range(12):
    resp = requests.get(f"{WORKSPACE_URL}/api/2.0/apps/{APP_NAME}", headers=headers)
    app_info = resp.json()
    state = app_info.get("app_status", {}).get("state", "UNKNOWN")
    url = app_info.get("url", "")
    print(f"[{i*10}s] App: {state}")
    if state == "RUNNING":
        print(f"\nApp running at: {url}")
        break
    time.sleep(10)
else:
    print(f"Timeout — check logs at: {url}/logz")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 — Troubleshooting
# MAGIC
# MAGIC | Problem | Cause | Solution |
# MAGIC |---------|-------|----------|
# MAGIC | App Not Available | Not listening on port 8000 | Check `command` in app.yaml |
# MAGIC | Lakebase connection error | Wrong credentials | Check env vars in app.yaml |
# MAGIC | `UndefinedColumn: name` | Missing column in table | Run notebook 01 again |
# MAGIC | `InsufficientPrivilege` | User is not table owner | Connect as owner and run GRANT ALL |
# MAGIC | Guardrail eval fails | Expired MLflow token | Check logs at /logz |
# MAGIC
# MAGIC **App logs**: Go to `<APP_URL>/logz`