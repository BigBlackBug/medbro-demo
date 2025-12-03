from app.ui.gradio_app import create_app
from config.logger import logger

# For direct execution
demo = create_app()

if __name__ == "__main__":
    logger.info("Starting Medical AI Assistant...")
    logger.info("Launching Gradio interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
