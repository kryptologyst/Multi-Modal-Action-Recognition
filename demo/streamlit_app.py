"""
Streamlit demo application for multi-modal action recognition.

Run with: streamlit run demo/streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from demo import create_streamlit_app
from src.config import DemoConfig

if __name__ == "__main__":
    config = DemoConfig()
    create_streamlit_app(config)
