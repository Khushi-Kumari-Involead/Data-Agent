import re
import duckdb
import pandas as pd
from typing import List, Dict
import os
from core.utils import sanitize_name, log

DB_PATH = "data/data.duckdb"

def initialize_db():
    """Initialize DuckDB connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH, read_only=False)

def register_df(df: pd.DataFrame, table: str, con: duckdb.DuckDBPyConnection):
    """Register a pandas DF as a DuckDB table and keep an in-memory copy."""
    df = df.dropna(how="all")
    # Preserve original column names from Excel/CSV, sanitizing only for DuckDB compatibility
    df.columns = [str(c).lower() if not pd.isna(c) and str(c).strip() else sanitize_name(str(c)) for c in df.columns]
    df = df.loc[:, ~(df.isna().all())]
    # Ensure unique column names
    seen = set()
    new_cols = []
    for i, col in enumerate(df.columns):
        if col in seen:
            new_cols.append(sanitize_name(f"{col}_{i}"))
        else:
            new_cols.append(col)
        seen.add(col)
    df.columns = new_cols
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.register("tmp_df", df)
    con.execute(f"CREATE TABLE {table} AS SELECT * FROM tmp_df")
    con.unregister("tmp_df")
    uploaded_dfs: Dict[str, pd.DataFrame] = {}  # table_name -> df
    uploaded_dfs[table] = df.copy()
    log(f"REGISTERED: {table} rows={len(df)} cols={list(df.columns)}")
    return uploaded_dfs

def list_tables(con: duckdb.DuckDBPyConnection) -> List[str]:
    try:
        return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    except Exception:
        return []

def get_schema(table: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    try:
        return con.execute(f"PRAGMA table_info('{table}')").df()
    except Exception:
        return pd.DataFrame()

def sample(table: str, con: duckdb.DuckDBPyConnection, n: int = 5) -> pd.DataFrame:
    try:
        return con.execute(f"SELECT * FROM {table} LIMIT {n}").df()
    except Exception:
        return pd.DataFrame()

def df_from_query(sql: str, con: duckdb.DuckDBPyConnection, limit: int = 5000) -> pd.DataFrame:
    ensure_select_sql(sql)
    if not re.search(r"\blimit\b", sql, re.I):
        sql = f"SELECT * FROM ({sql}) AS q LIMIT {limit}"
    return con.execute(sql).df()

def ensure_select_sql(sql: str):
    SAFE_SQL_PREFIXES = ("select", "with")
    if not sql or not sql.strip().lower().startswith(SAFE_SQL_PREFIXES):
        raise ValueError("Unsafe SQL: only SELECT/CTE queries are permitted.")

def tables_context(con: duckdb.DuckDBPyConnection) -> str:
    """Generate a context string for available database tables."""
    tbls = list_tables(con)
    if not tbls:
        return "(no tables loaded)"
    lines = []
    for t in tbls:
        try:
            sch = get_schema(t, con)
            cols = ", ".join(f"{r['name']}:{r['type']}" for _, r in sch.iterrows())
            sample_data = sample(t, con, n=3).to_dict(orient="records")
            sample_str = "; Sample: " + str(sample_data)[:100] if sample_data else ""
        except Exception:
            cols = "(schema error)"
            sample_str = ""
        lines.append(f"- {t} -> {cols}{sample_str}")
    return "TABLES:\n" + "\n".join(lines)

def clear_tables(con: duckdb.DuckDBPyConnection):
    """Drop all tables in the DuckDB database."""
    try:
        tables = list_tables(con)
        for table in tables:
            con.execute(f"DROP TABLE {table}")
        log("Cleared all tables from DuckDB database.")
    except Exception as e:
        log(f"CLEAR TABLES ERROR: {e}")