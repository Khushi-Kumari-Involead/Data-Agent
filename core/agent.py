import pandas as pd
from collections import deque
import json
import re
from typing import Tuple, List, Dict, Any, Optional
from langchain_ollama import ChatOllama
import matplotlib.pyplot as plt
from core.utils import log, try_markdown, persist_memory
from core.db import list_tables, get_schema, df_from_query, tables_context
from core.charts import make_chart_from_df
from core.actions import ACTIONS_SPEC, run_python_safely, make_full_user_code
import time
from langchain_core.exceptions import LangChainException

llm = ChatOllama(model="mistral", temperature=0, timeout=30)

def run_planner(context: str, query: str, short_memory_list: List[Dict[str,str]]) -> Dict[str,Any]:
    """Run the LLM planner to generate an action plan."""
    messages = [
        {"role": "system", "content": ACTIONS_SPEC},
        {"role": "system", "content": f"CONTEXT:\n{context}"},
    ]
    for turn in short_memory_list:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["ai"]})
    messages.append({"role": "user", "content": query})
    
    try:
        start_time = time.time()
        raw = llm.invoke(messages).content
        log(f"LLM response time: {time.time() - start_time:.2f} seconds")
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.strip()
        s = raw.find("{")
        e = raw.rfind("}")
        if s == -1 or e == -1 or e <= s:
            error_msg = f"Planner did not return valid JSON. Raw output:\n{raw}"
            log(error_msg)
            raise ValueError(error_msg)
        json_text = raw[s:e+1]
        try:
            plan = json.loads(json_text)
            log(f"Parsed plan: {json.dumps(plan)[:500]}")
            return plan
        except json.JSONDecodeError as exc:
            error_msg = f"Planner JSON parse error: {exc}\nRaw JSON text:\n{json_text}"
            log(error_msg)
            raise ValueError(error_msg)
    except LangChainException as e:
        error_msg = f"LLM invocation failed: {e}"
        log(error_msg)
        raise
    except Exception as e:
        error_msg = f"Unexpected planner error: {e}"
        log(error_msg)
        raise

def rewrite_json_builder_sql(sql: str) -> str:
    """Rewrite common json_build_object Postgres pattern to a simpler SELECT fallback."""
    if "json_build_object" not in sql.lower():
        return sql
    m = re.search(r"select\s+json_build_object\s*\(.*?\)\s+from\s+([A-Za-z0-9_]+)", sql, flags=re.I | re.S)
    if m:
        alias = m.group(1)
        new_sql = re.sub(r"select\s+json_build_object\s*\(.*?\)\s+from\s+[A-Za-z0-9_]+", f"select * from {alias}", sql, flags=re.I | re.S)
        return new_sql
    return re.sub(r"select\s+json_build_object\s*\(.*?\)", "select *", sql, flags=re.I | re.S)

def select_default_columns(schema_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Select default x (categorical) and y (numeric) columns from schema."""
    categorical_types = ["VARCHAR", "TEXT", "STRING"]
    numeric_types = ["INTEGER", "FLOAT", "DOUBLE", "DECIMAL"]
    
    x_col = None
    y_col = None
    
    for _, row in schema_df.iterrows():
        col_name = row["name"]
        col_type = row["type"]
        # Treat columns with numeric names as numeric if their type is numeric
        if not y_col and (col_type in numeric_types or (col_name.replace(".", "").isdigit() and col_type in ["FLOAT", "DOUBLE"])):
            y_col = col_name
        # Prefer non-numeric names for categorical columns
        if not x_col and (col_type in categorical_types or not col_name.replace(".", "").isdigit()):
            x_col = col_name
        if x_col and y_col:
            break
    
    return x_col, y_col

def run_agent(user_query: str, short_memory_deque: deque, con: Any, uploaded_dfs: Dict[str, pd.DataFrame], db_path: str, max_steps: int = 3) -> Tuple[List[Dict[str,Any]], str, Optional[plt.Figure], Optional[str]]:
    """Run the agent to process a user query."""
    try:
        ctx = tables_context(con)
        log(f"Generated context: {ctx[:500]}")
    except Exception as e:
        error_msg = f"Failed to generate table context: {e}"
        log(error_msg)
        return [{"action": "context_error", "error": str(e)}], error_msg, None, None

    steps: List[Dict[str,Any]] = []
    last_answer = ""
    returned_fig = None
    returned_path = None

    for step_i in range(max_steps):
        try:
            plan = run_planner(ctx, user_query, list(short_memory_deque))
        except Exception as e:
            last_answer = f"(planner error) {e}"
            steps.append({"action": "planner_error", "error": str(e)})
            log(f"PLANNER ERROR: {e}")
            break

        action = plan.get("action", "")
        args = plan.get("args", {}) or {}
        log(f"PLAN step {step_i}: {json.dumps(plan)[:500]}")

        try:
            if action == "list_tables":
                tbls = list_tables(con)
                last_answer = "Available tables: " + (", ".join(tbls) if tbls else "(none)")
                steps.append({"action": "list_tables", "result": last_answer})
                break

            elif action == "list_columns":
                table = args.get("table", "").strip()
                if not table:
                    last_answer = "No table provided for list_columns."
                    steps.append({"action": "list_columns", "error": last_answer})
                    break
                schema_df = get_schema(table, con)
                last_answer = try_markdown(schema_df)
                steps.append({"action": "list_columns", "table": table, "schema": schema_df.to_dict(orient="records")})
                break

            elif action == "sql":
                sql = (args.get("sql") or "").strip()
                if not sql:
                    last_answer = "(sql error) empty SQL"
                    steps.append({"action": "sql", "error": last_answer})
                    break
                try:
                    safe_sql = rewrite_json_builder_sql(sql)
                    df = df_from_query(safe_sql, con, limit=10000)
                    last_answer = f"Rows: {len(df)}\nSample:\n{try_markdown(df.head(10))}"
                    steps.append({"action": "sql", "sql": safe_sql, "rows": len(df), "sample": df.head(5).to_dict(orient="records")})
                except Exception as e:
                    last_answer = f"(sql error) {e}"
                    steps.append({"action": "sql", "error": str(e), "sql": sql})
                break

            elif action == "chart":
                table = args.get("table", "").strip()
                if not table:
                    last_answer = "Chart action requires 'table' in args. Please upload a dataset and try again."
                    steps.append({"action": "chart", "error": last_answer})
                    break
                x = args.get("x")
                y = args.get("y")
                ctype = (args.get("type") or "line").lower()
                agg = (args.get("agg") or "none").lower()
                where = args.get("filter") or ""

                # Validate table and columns
                schema_df = get_schema(table, con)
                if schema_df.empty:
                    last_answer = f"Table '{table}' not found. Please upload a dataset and try again."
                    steps.append({"action": "chart", "error": last_answer})
                    break
                
                valid_columns = schema_df["name"].tolist()
                
                # Handle vague queries by selecting default columns
                if not x or not y or x not in valid_columns or y not in valid_columns:
                    default_x, default_y = select_default_columns(schema_df)
                    if not default_x or not default_y:
                        last_answer = (
                            f"Cannot generate chart: No valid categorical (x) or numeric (y) columns found in '{table}'. "
                            f"Available columns: {', '.join(valid_columns)}. "
                            "Please upload a dataset with proper column headers (e.g., strings for categories, numbers for values)."
                        )
                        steps.append({"action": "chart", "error": last_answer})
                        break
                    x = default_x
                    y = default_y
                    log(f"Using default columns for chart: x={x}, y={y}")

                # Construct SQL query
                base_sql = f"SELECT * FROM {table}"
                if where.strip():
                    base_sql += f" WHERE {where}"
                if agg in {"sum", "mean", "avg", "min", "max"}:
                    func = "AVG" if agg in {"mean", "avg"} else agg.upper()
                    sql = f"SELECT {x} AS x, {func}({y}) AS y FROM ({base_sql}) GROUP BY {x} ORDER BY x"
                else:
                    sql = f"SELECT {x} AS x, {y} AS y FROM ({base_sql})"
                
                try:
                    df = df_from_query(sql, con, limit=10000)
                    if df.empty:
                        last_answer = "No data to plot."
                        steps.append({"action": "chart", "sql": sql, "result": last_answer})
                    else:
                        fig, path = make_chart_from_df(df, ctype, "x", "y")
                        returned_fig, returned_path = fig, path
                        last_answer = f"Chart generated and saved to {path}"
                        steps.append({"action": "chart", "sql": sql, "image": path})
                    break
                except Exception as e:
                    last_answer = f"(chart error) {e}"
                    steps.append({"action": "chart", "error": str(e), "sql": sql})
                    break

            elif action == "python":
                code = args.get("code") or ""
                if not code:
                    last_answer = "(python error) empty code"
                    steps.append({"action": "python", "error": last_answer})
                    break
                try:
                    table_match = re.search(r"con\.execute\s*\(\s*['\"].*?FROM\s+([A-Za-z0-9_]+)", code, re.I)
                    table_name = table_match.group(1) if table_match else "dataset"
                    out = run_python_safely(code, con, uploaded_dfs)
                    full_code = make_full_user_code(code, table_name)
                    steps.append({"action": "python", "code": full_code, "stdout": out})
                    last_answer = f"```python\n{full_code}\n```\n\nOutput:\n{out}"
                except Exception as e:
                    last_answer = f"(python execution error) {e}"
                    steps.append({"action": "python", "error": str(e), "code": code})
                break

            elif action == "final":
                answer_text = args.get("answer") or ""
                last_answer = answer_text
                steps.append({"action": "final", "result": answer_text})
                break

            else:
                last_answer = "I could not choose a valid action."
                steps.append({"action": "final", "result": last_answer})
                break

        except Exception as e:
            last_answer = f"(action error) {e}"
            steps.append({"action": "action_error", "error": str(e)})
            log(f"ACTION ERROR: {e}")
            break

    try:
        short_memory_deque.append({"user": user_query, "ai": last_answer})
        while len(short_memory_deque) > 2:
            short_memory_deque.popleft()
        persist_memory(user_query, last_answer)
    except Exception as e:
        log(f"MEMORY UPDATE ERROR: {e}")

    log(f"FINAL ANSWER: {last_answer} -- steps: {json.dumps(steps, default=str)[:1500]}")
    return steps, last_answer, returned_fig, returned_path