# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Lakebase Setup
# MAGIC
# MAGIC This notebook creates all tables and the application user in Lakebase.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Have a Lakebase instance provisioned (Compute > Lakebase)
# MAGIC 2. Click **Connect** on your instance and copy the **Connection string**
# MAGIC
# MAGIC ## Instructions
# MAGIC 1. Run cell 1.1 first (installs dependencies and restarts Python)
# MAGIC 2. Fill in the configuration in cell 1.2
# MAGIC 3. Run all remaining cells

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 — Install dependencies (run this first!)

# COMMAND ----------

# MAGIC %pip install "psycopg[binary]" "databricks-sdk>=0.89.0" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 — Configuration
# MAGIC
# MAGIC After Python restarts, fill in and run this cell.

# COMMAND ----------

# ============================================================
# FILL IN WITH YOUR DATA
# ============================================================
# 1) Go to Compute > Lakebase > click your instance > Connect
# 2) Click "Copy snippet" and paste below:
LAKEBASE_CONNECTION = "<PASTE_CONNECTION_STRING>"  # postgresql://user@ep-xxx.database...

# 3) Project name (visible in the Lakebase sidebar, e.g. "llm-caching")
LAKEBASE_PROJECT = "llm-caching"

# App user credentials (used by the application to connect)
APP_USER = "cache_app"
APP_PASSWORD = "<YOUR_PASSWORD>"  # e.g. MyStr0ngP@ss2026!

# --- Parse connection string ---
from urllib.parse import urlparse, unquote
if LAKEBASE_CONNECTION.startswith("postgresql://") or LAKEBASE_CONNECTION.startswith("postgres://"):
    _parsed = urlparse(LAKEBASE_CONNECTION)
    LAKEBASE_HOST = _parsed.hostname
    _owner_email = unquote(_parsed.username) if _parsed.username else None
else:
    LAKEBASE_HOST = LAKEBASE_CONNECTION.strip()
    _owner_email = None
print(f"Host: {LAKEBASE_HOST}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 — Connect as owner (your Databricks user)
# MAGIC
# MAGIC Generates an OAuth token automatically via SDK (no hardcoded tokens).

# COMMAND ----------

import psycopg
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get owner email
if not _owner_email:
    _owner_email = w.current_user.me().user_name
print(f"Connecting as: {_owner_email}")

# Auto-discover endpoint and generate OAuth token
_branches = list(w.postgres.list_branches(parent=f"projects/{LAKEBASE_PROJECT}"))
if not _branches:
    raise RuntimeError(f"No branches found for project '{LAKEBASE_PROJECT}'. Check the project name.")
_branch_name = _branches[0].name
print(f"Branch: {_branch_name}")

_endpoints = list(w.postgres.list_endpoints(parent=_branch_name))
if not _endpoints:
    raise RuntimeError(f"No endpoints found in branch '{_branch_name}'.")
_endpoint_name = _endpoints[0].name
print(f"Endpoint: {_endpoint_name}")

cred = w.postgres.generate_database_credential(endpoint=_endpoint_name)
_token = cred.token
print(f"OAuth token generated (length: {len(_token)})")

conn_owner = psycopg.connect(
    host=LAKEBASE_HOST,
    port=5432,
    dbname="databricks_postgres",
    user=_owner_email,
    password=_token,
    sslmode="require",
    autocommit=True,
)
print("Connected to Lakebase as owner!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 — Create application user

# COMMAND ----------

cur = conn_owner.cursor()
try:
    cur.execute(f"CREATE USER {APP_USER} WITH PASSWORD '{APP_PASSWORD}'")
    print(f"User '{APP_USER}' created!")
except Exception as e:
    if "already exists" in str(e):
        print(f"User '{APP_USER}' already exists, OK.")
    else:
        raise e
cur.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 — Create tables

# COMMAND ----------

cur = conn_owner.cursor()

# Products
cur.execute("""
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10,2) NOT NULL,
    stock INTEGER NOT NULL DEFAULT 0,
    description TEXT DEFAULT '',
    sku TEXT DEFAULT ''
)
""")
print("Table 'products' created.")

# Sales
cur.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    customer_name TEXT NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price NUMERIC(10,2) NOT NULL,
    total NUMERIC(10,2) NOT NULL,
    sale_date TIMESTAMP DEFAULT NOW()
)
""")
print("Table 'sales' created.")

# LLM Cache
cur.execute("""
CREATE TABLE IF NOT EXISTS llm_cache (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    embedding TEXT,
    model TEXT DEFAULT '',
    tokens_used INTEGER DEFAULT 0,
    response_time_ms INTEGER DEFAULT 0,
    query_hash TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_hit_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
print("Table 'llm_cache' created.")

# Conversations
cur.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    cached BOOLEAN DEFAULT FALSE,
    response_time_ms INTEGER DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE,
    similarity FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
print("Table 'conversations' created.")

# Guardrails
cur.execute("""
CREATE TABLE IF NOT EXISTS guardrails (
    id SERIAL PRIMARY KEY,
    name TEXT DEFAULT '',
    rule_text TEXT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
print("Table 'guardrails' created.")

# Guardrail evaluations
cur.execute("""
CREATE TABLE IF NOT EXISTS guardrail_evaluations (
    id SERIAL PRIMARY KEY,
    user_message TEXT,
    agent_response TEXT,
    overall_pass BOOLEAN,
    pass_rate FLOAT,
    evaluations_json TEXT,
    summary TEXT,
    eval_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
print("Table 'guardrail_evaluations' created.")

cur.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.6 — Grant permissions to app user

# COMMAND ----------

cur = conn_owner.cursor()
cur.execute(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {APP_USER}")
cur.execute(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {APP_USER}")
cur.close()
print(f"Permissions granted to '{APP_USER}'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.7 — Insert sample products

# COMMAND ----------

cur = conn_owner.cursor()

# Check if products already exist
cur.execute("SELECT COUNT(*) FROM products")
count = cur.fetchone()[0]

if count == 0:
    cur.execute("""
    INSERT INTO products (name, category, price, stock, description, sku) VALUES
    ('MacBook Pro 14"', 'Notebooks', 12499.00, 15, 'Apple M3 Pro, 18GB RAM, 512GB SSD', 'MBP14-M3'),
    ('MacBook Air 13"', 'Notebooks', 8999.00, 20, 'Apple M3, 8GB RAM, 256GB SSD', 'MBA13-M3'),
    ('Dell XPS 15', 'Notebooks', 9499.00, 12, 'Intel i7, 16GB RAM, 512GB SSD', 'DXPS15'),
    ('ThinkPad X1 Carbon', 'Notebooks', 8799.00, 10, 'Intel i7, 16GB RAM, 512GB SSD, 14"', 'TPX1C'),
    ('iPhone 15 Pro', 'Smartphones', 7999.00, 25, 'A17 Pro, 256GB, Natural Titanium', 'IP15P-256'),
    ('iPhone 15', 'Smartphones', 5999.00, 30, 'A16, 128GB', 'IP15-128'),
    ('Samsung Galaxy S24 Ultra', 'Smartphones', 7499.00, 18, 'Snapdragon 8 Gen 3, 256GB, S Pen', 'SGS24U'),
    ('Samsung Galaxy S24', 'Smartphones', 4999.00, 22, 'Exynos 2400, 128GB', 'SGS24'),
    ('iPad Pro 11"', 'Tablets', 8499.00, 14, 'M2, 128GB, Wi-Fi', 'IPADP11'),
    ('iPad Air', 'Tablets', 4999.00, 20, 'M1, 64GB, Wi-Fi', 'IPADA'),
    ('AirPods Pro 2', 'Accessories', 1899.00, 40, 'USB-C, ANC, Adaptive Audio', 'APP2'),
    ('Magic Keyboard', 'Accessories', 1099.00, 25, 'Touch ID, ABNT2 layout', 'MK-ABNT'),
    ('Samsung Galaxy Tab S9', 'Tablets', 5499.00, 16, 'Snapdragon 8 Gen 2, 128GB', 'SGTS9'),
    ('Apple Watch Series 9', 'Wearables', 3299.00, 20, 'GPS, 45mm, Aluminum', 'AW9-45'),
    ('Galaxy Watch 6', 'Wearables', 2199.00, 18, 'BT, 44mm', 'GW6-44')
    """)
    print("15 sample products inserted!")
else:
    print(f"Already have {count} products. Skipping insert.")

cur.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.8 — Verify everything

# COMMAND ----------

# Test connection with app user
conn_app = psycopg.connect(
    host=LAKEBASE_HOST,
    port=5432,
    dbname="databricks_postgres",
    user=APP_USER,
    password=APP_PASSWORD,
    sslmode="require",
    autocommit=True,
)

cur = conn_app.cursor()
cur.execute("SELECT COUNT(*) FROM products")
print(f"Products: {cur.fetchone()[0]}")

cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'guardrails' ORDER BY ordinal_position")
print(f"Guardrails columns: {[r[0] for r in cur.fetchall()]}")

cur.close()
conn_app.close()
conn_owner.close()

print("\n=== SETUP COMPLETE ===")
print(f"Host: {LAKEBASE_HOST}")
print(f"User: {APP_USER}")
print(f"Password: {APP_PASSWORD}")
print("Use these values in app.yaml (Step 3)")