"""
Gradio demo application for multi-modal action recognition.

Run with: python demo/gradio_app.py
"""

import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from demo import create_gradio_app
from src.config import DemoConfig

if __name__ == "__main__":
    config = DemoConfig()
    interface = create_gradio_app(config)
    interface.launch(share=False)
