import uuid

from app.core.models import AnalysisResult
from app.services.llm import get_llm_provider
from app.services.stt import get_stt_provider
from app.services.tts import get_tts_provider
from config.logger import logger
from config.prompts import get_analysis_prompt
from config.settings import config


class MedicalSessionService:
    def __init__(self):
        self.stt = get_stt_provider()
        self.llm = get_llm_provider()
        self.tts = get_tts_provider()
        logger.info(f"MedicalSessionService initialized. Use Mock: {config.USE_MOCK_SERVICES}")

    async def process_audio(self, audio_path: str) -> tuple[str, AnalysisResult]:
        logger.info(f"Processing audio file: {audio_path}")

        # 1. STT
        logger.info("Starting STT transcription...")
        transcript = await self.stt.transcribe(audio_path)
        logger.info(f"STT completed. Transcript length: {len(transcript)} chars")

        # 2. LLM Analysis
        logger.info("Starting LLM analysis...")
        system_prompt = get_analysis_prompt()
        analysis = await self.llm.analyze(transcript, system_prompt)
        logger.info("LLM analysis completed successfully.")

        return transcript, analysis

    async def generate_voice_recommendations(self, recommendations: list[str]) -> str:
        logger.info(f"Generating voice recommendations for {len(recommendations)} items.")
        text_to_speak = "Рекомендации по назначению: " + ". ".join(recommendations)
        filename = f"recs_{uuid.uuid4()}.mp3"
        output_path = config.TEMP_DIR / filename

        audio_file = await self.tts.speak(text_to_speak, str(output_path))
        logger.info(f"Audio generated at: {audio_file}")
        return audio_file


# Singleton or Factory
def get_session_service() -> MedicalSessionService:
    return MedicalSessionService()
