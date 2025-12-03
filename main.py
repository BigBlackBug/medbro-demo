from app.ui.gradio_app import create_app
from config.logger import logger

# For direct execution
if __name__ == "__main__":
    logger.info("Starting Medical AI Assistant...")
    app = create_app()
    logger.info("Launching Gradio interface...")
    app.launch(server_name="0.0.0.0", server_port=7860)
