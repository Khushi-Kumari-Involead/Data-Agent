import streamlit as st
from ui.components import render_ui
from core.db import initialize_db
from core.utils import log

def main():
    try:
        db_connection = initialize_db()
        render_ui(db_connection)
    except Exception as e:
        st.error(f"Application error: {e}")
        log(f"APP ERROR: {e}")

if __name__ == "__main__":
    main()