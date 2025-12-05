import os

from app.ui.gradio_app_streaming import create_streaming_app
from config.logger import logger

demo = create_streaming_app()

if __name__ == "__main__":
    logger.info("Starting Medical AI Assistant (Streaming Mode)...")
    logger.info("Launching Gradio interface...")
    port = int(os.getenv("PORT", 7861))
    demo.launch(server_name="0.0.0.0", server_port=port)

