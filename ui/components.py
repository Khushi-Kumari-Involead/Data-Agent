import streamlit as st
from typing import Dict, Any
from core.db import list_tables, get_schema, sample, register_df, clear_tables
from core.utils import log, coerce_first_row_to_header, split_blocks_by_empty_rows, sanitize_name, save_chat_session, load_chat_names
from core.agent import run_agent
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import deque
import json

def render_ui(con):
    st.set_page_config(layout="wide", page_title="Data Agent")
    from core.utils import initialize_storage
    initialize_storage()

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "recent_charts" not in st.session_state:
        st.session_state.recent_charts = []
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "short_memory" not in st.session_state:
        st.session_state.short_memory = deque(maxlen=2)
    if "show_new_chat_dialog" not in st.session_state:
        st.session_state.show_new_chat_dialog = False
    if "chat_name" not in st.session_state:
        st.session_state.chat_name = ""
    if "show_file_uploader" not in st.session_state:
        st.session_state.show_file_uploader = False

    db_path = "data/data.duckdb"
    uploaded_dfs: Dict[str, pd.DataFrame] = {}

    # Sidebar
    with st.sidebar:
        st.header("Data Agent")
        
        # New Chat Button
        if st.button("New Chat"):
            st.session_state.show_new_chat_dialog = True

        # New Chat Dialog
        if st.session_state.show_new_chat_dialog:
            with st.container(border=True):
                st.subheader("Save Current Chat")
                chat_name = st.text_input("Enter a name for the current chat session:", key="chat_name_input")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save and Clear"):
                        if chat_name.strip():
                            save_chat_session(chat_name)
                            clear_tables(con)
                            # Clear chat history and charts
                            st.session_state.chat_history = []
                            st.session_state.recent_charts = []
                            st.session_state.short_memory = deque(maxlen=2)
                            st.session_state.uploaded_file = None
                            st.session_state.show_new_chat_dialog = False
                            st.session_state.show_file_uploader = False
                            # Clear memory file
                            open("data/memory.txt", "w").close()
                            # Clear charts directory
                            for f in os.listdir("charts"):
                                os.remove(os.path.join("charts", f))
                            st.rerun()
                        else:
                            st.error("Please enter a chat name.")
                with col2:
                    if st.button("Cancel"):
                        st.session_state.show_new_chat_dialog = False

        # Display saved chat names
        st.subheader("Saved Chats")
        chat_names = load_chat_names()
        if chat_names:
            for name in chat_names:
                st.write(name)
        else:
            st.write("(No saved chats)")

        # Table selection
        tables = list_tables(con)
        selected_table = st.selectbox("Select Table", [""] + tables)
        show_schema = False
        if selected_table:
            show_schema = st.checkbox("Show Schema", value=True)
            if show_schema:
                schema_df = get_schema(selected_table, con)
                st.write("**Schema**")
                st.dataframe(schema_df[["name", "type"]], use_container_width=True)
            st.write("**Preview**")
            st.dataframe(sample(selected_table, con, n=5), use_container_width=True)

    # Main content
    st.header("Chat")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("image"):
                try:
                    st.image(msg["image"])
                except Exception as e:
                    log(f"IMAGE DISPLAY ERROR: {e}")

    # Recent charts
    if st.session_state.recent_charts:
        st.header("Recent Charts")
        for path in reversed(st.session_state.recent_charts[-3:]):
            try:
                st.image(path)
            except Exception as e:
                log(f"CHART DISPLAY ERROR: {e}")

    # Chat input with attach button and file uploader at the bottom
    with st.container():
        st.markdown("---")  # Separator for visual clarity
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("📎", help="Attach a file"):
                st.session_state.show_file_uploader = not st.session_state.show_file_uploader
        with col2:
            prompt = st.chat_input("Ask about your data...")

        if st.session_state.show_file_uploader:
            upload_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
            if upload_file is not None and upload_file != st.session_state.uploaded_file:
                st.session_state.uploaded_file = upload_file
                try:
                    if upload_file.name.endswith(".csv"):
                        df = pd.read_csv(upload_file)
                        base_name = sanitize_name(os.path.splitext(upload_file.name)[0])
                        uploaded_dfs.update(register_df(df, base_name, con))
                    else:
                        xls = pd.ExcelFile(upload_file)
                        for sheet in xls.sheet_names:
                            df = pd.read_excel(upload_file, sheet_name=sheet)
                            blocks = split_blocks_by_empty_rows(df)
                            for block, start_idx in blocks:
                                block = coerce_first_row_to_header(block)
                                base_name = sanitize_name(f"{os.path.splitext(upload_file.name)[0]}_{sheet}_{start_idx}")
                                uploaded_dfs.update(register_df(block, base_name, con))
                    st.session_state.chat_history.append({"role": "system", "content": f"Uploaded: {upload_file.name}"})
                    st.session_state.show_file_uploader = False  # Hide uploader after successful upload
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    log(f"UPLOAD ERROR: {e}")

    # Process user input
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    steps, answer, fig, path = run_agent(prompt, st.session_state.short_memory, con, uploaded_dfs, db_path)
                    st.write(answer)
                    log(f"Steps: {json.dumps(steps, default=str)}")
                    if path and fig:
                        plt.close(fig)
                        st.image(path)
                        st.session_state.recent_charts.append(path)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer, "image": path})
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    if steps and steps[-1].get("action") == "python":
                        st.rerun()
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    log(f"AGENT ERROR: {e}")
