import os
import re
from datetime import datetime
import pandas as pd
from typing import Optional, List, Tuple

MEMORY_FILE = "data/memory.txt"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
CHAT_SESSIONS_FILE = "data/saved_chats.txt"

def initialize_storage():
    """Create necessary directories for storage."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    os.makedirs("scratch", exist_ok=True)
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(CHAT_SESSIONS_FILE), exist_ok=True)

def log(msg: str):
    """Log a message to the log file."""
    ts = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def try_markdown(df: pd.DataFrame) -> str:
    """Return a textual representation: prefer markdown table, fallback to plain string."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.head(10).to_string(index=False)

def persist_memory(user_msg: str, ai_msg: str, memory: Optional[object] = None):
    """Persist conversation to memory file and update LangChain memory if available."""
    try:
        ts = datetime.now().isoformat()
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] Q: {user_msg}\nA: {ai_msg}\n\n")
    except Exception as e:
        log(f"MEMORY WRITE ERROR: {e}")
    if memory is not None:
        try:
            memory.chat_memory.add_user_message(user_msg)
            memory.chat_memory.add_ai_message(ai_msg)
        except Exception as e:
            log(f"LANGCHAIN MEMORY ERROR: {e}")

def sanitize_name(name: str) -> str:
    """Sanitize a name for use as a table or column name, preserving original as much as possible."""
    if pd.isna(name) or not str(name).strip():
        return "unnamed"
    # Convert to string and preserve original content
    name_str = str(name).strip()
    # Replace invalid characters (except letters, numbers, underscores) with underscores
    base = re.sub(r"[^A-Za-z0-9_]", "_", name_str)
    # Ensure the name doesn't start with a digit
    if re.match(r"^\d", base):
        base = f"name_{base}"
    # Remove multiple consecutive underscores
    base = re.sub(r"_+", "_", base).strip("_")
    # Ensure non-empty name
    return base.lower() or "unnamed"

def split_blocks_by_empty_rows(df: pd.DataFrame) -> List[Tuple[pd.DataFrame, int]]:
    """Split a sheet into blocks separated by fully-empty rows."""
    blocks = []
    current_rows = []
    start_idx = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if row.isna().all():
            if current_rows:
                block = pd.DataFrame(current_rows).reset_index(drop=True)
                blocks.append((block, start_idx))
                current_rows = []
            start_idx = i + 1
        else:
            current_rows.append(row)
    if current_rows:
        block = pd.DataFrame(current_rows).reset_index(drop=True)
        blocks.append((block, start_idx))
    return blocks

def coerce_first_row_to_header(block: pd.DataFrame) -> pd.DataFrame:
    """Use first row as header if it exists and is not entirely empty; otherwise assign default column names."""
    block = block.copy()
    if block.empty:
        return block
    # Use pandas' default headers (from Excel/CSV) directly, assuming they are the column names
    if block.columns.tolist() != [f"Unnamed: {i}" for i in range(block.shape[1])]:
        # Excel/CSV already provided headers; sanitize them
        block.columns = [sanitize_name(str(c)) for c in block.columns]
        return block
    # If no headers provided (e.g., all "Unnamed: i"), use first row as headers
    first = block.iloc[0]
    non_null_count = first.notna().sum()
    if non_null_count > 0:
        new_cols = [sanitize_name(str(c)) if not pd.isna(c) and str(c).strip() else f"unnamed_{i}" for i, c in enumerate(first.tolist())]
        block.columns = new_cols
        block = block.iloc[1:].reset_index(drop=True)
    else:
        # If first row is empty, assign default names
        block.columns = [f"unnamed_{i}" for i in range(block.shape[1])]
    return block

def save_chat_session(chat_name: str):
    """Save the chat session name to a file."""
    try:
        with open(CHAT_SESSIONS_FILE, "a", encoding="utf-8") as f:
            ts = datetime.now().isoformat()
            f.write(f"[{ts}] {chat_name}\n")
        log(f"Saved chat session: {chat_name}")
    except Exception as e:
        log(f"CHAT SESSION SAVE ERROR: {e}")

def load_chat_names() -> List[str]:
    """Load the list of saved chat session names."""
    try:
        if os.path.exists(CHAT_SESSIONS_FILE):
            with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return [line.strip().split("] ")[1] for line in f if "] " in line]
        return []
    except Exception as e:
        log(f"CHAT SESSION LOAD ERROR: {e}")
        return []