import io
import re
import contextlib
import pandas as pd
import numpy as np
from typing import Dict, Any
import duckdb

ACTIONS_SPEC = """
You are a precise data analyst assistant. You MUST respond ONLY with a single JSON object (no prose).
Schema:
{
  "action": "list_tables | list_columns | sql | python | chart | final",
  "args": { ... }
}
Examples:
- List tables: {"action":"list_tables","args":{}}
- List columns: {"action":"list_columns","args":{"table":"table_name"}}
- SQL query: {"action":"sql","args":{"sql":"SELECT AVG(Salary) AS avg_salary FROM table_name"}}
- Python (no imports): {"action":"python","args":{"code":"df = con.execute('SELECT * FROM table_name').df(); print(df['Salary'].mean())"}}
- Chart: {"action":"chart","args":{"table":"table_name","type":"line","x":"YearsExperience","y":"Salary","agg":"none"}}
- Final: {"action":"final","args":{"answer":"Short final natural language answer"}}

Rules:
- Use only table names shown in CONTEXT.
- Prefer SQL for aggregates (AVG, SUM, MIN, MAX). Use Python for median, describe(), correlation, advanced transforms.
- For Python code returned to be executed at runtime, DO NOT include import statements (execution environment already has pd, np, con, uploaded_dfs).
- When providing Python code for the user to copy/paste, include the full imports (we will show a full code snippet).
- Always return a valid single JSON object and nothing else.
"""

def preprocess_python_code(code: str) -> str:
    """Prepare Python code for safe execution."""
    code = re.sub(r"duckdb\.query\s*\(", "con.execute(", code, flags=re.I)
    code = code.replace(".to_df()", ".df()")
    code = re.sub(r"(?m)^\s*import\s+[^\n]+$", "", code)
    code = re.sub(r"(?m)^\s*from\s+[^\n]+\s+import\s+[^\n]+$", "", code)
    return code.strip()

def make_full_user_code(code: str, table_name: str) -> str:
    """Create a user-friendly Python snippet with imports for a given table."""
    header = (
        "import pandas as pd\n"
        "\n"
        "# Specify the path to your dataset file (CSV or Excel)\n"
        "file_path = 'path/to/your/dataset.csv'  # Replace with your file path\n"
        "\n"
        "# Load the dataset\n"
        "try:\n"
        "    if file_path.endswith('.csv'):\n"
        "        df = pd.read_csv(file_path)\n"
        "    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):\n"
        "        df = pd.read_excel(file_path)\n"
        "    else:\n"
        "        raise ValueError('Unsupported file format. Use CSV or Excel (.xlsx, .xls)')\n"
        "except Exception as e:\n"
        "    print(f'Error loading file: {e}')\n"
        "    exit(1)\n"
        "\n"
        f"# Using table: {table_name}\n"
    )
    # Replace 'con.execute(...).df()' with 'df' in the code
    code = re.sub(r"con\.execute\s*\(\s*['\"](.*?)['\"]\s*\)\.df\s*\(\s*\)", "df", code)
    return header + code.strip()

def run_python_safely(code: str, con: Any, uploaded_dfs: Dict[str, pd.DataFrame]) -> str:
    """Execute Python code in a restricted sandbox."""
    exec_code = preprocess_python_code(code)
    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum, "print": print,
        "str": str, "int": int, "float": float, "dict": dict, "list": list,
    }
    allowed_globals = {
        "__builtins__": safe_builtins,
        "con": con,
        "pd": pd,
        "np": np,
        "duckdb": duckdb,
        "uploaded_dfs": uploaded_dfs,
    }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            local_env: Dict[str, Any] = {}
            exec(exec_code, allowed_globals, local_env)
        out = buf.getvalue().strip()
        for v in reversed(list(local_env.values())):
            if isinstance(v, pd.DataFrame):
                from core.utils import try_markdown
                return try_markdown(v.head(20))
        return out or "(python executed with no stdout)"
    except Exception as e:
        return f"(python error) {e}"
