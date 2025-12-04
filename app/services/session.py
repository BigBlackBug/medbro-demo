import uuid
from app.core.models import AnalysisResult, DialogueTurn, GeneratedDialogue
from app.services.llm import get_llm_provider
from app.services.stt import get_stt_provider
from app.services.tts import get_tts_provider
from config.logger import logger
from config.prompts import get_analysis_prompt, SYSTEM_PROMPT_GENERATE_DIALOGUE
from config.settings import config


class MedicalSessionService:
    def __init__(self):
        self.stt = get_stt_provider()
        self.llm = get_llm_provider()
        self.tts = get_tts_provider()
        logger.info(f"MedicalSessionService initialized. Use Mock: {config.USE_MOCK_SERVICES}")

    async def process_audio(self, audio_path: str) -> tuple[str | list[DialogueTurn], AnalysisResult]:
        logger.info(f"Processing audio file: {audio_path}")

        # 1. STT
        logger.info("Starting STT transcription...")
        transcript: str = await self.stt.transcribe_raw(audio_path)
        logger.info(f"STT completed. Transcript length: {len(transcript)}")

        # 2. LLM Analysis
        logger.info("Starting LLM analysis...")
        system_prompt: str = get_analysis_prompt()
        analysis: AnalysisResult = await self.llm.analyze_raw(transcript, system_prompt)
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

    async def generate_and_analyze_sample(self) -> tuple[str | list[DialogueTurn], AnalysisResult]:
        logger.info("Generating sample dialogue...")
        
        # 1. Generate dialogue text
        generated_dialogue: GeneratedDialogue = await self.llm.generate_dialogue(SYSTEM_PROMPT_GENERATE_DIALOGUE)
        
        # 2. Convert to DialogueTurn list
        transcript_turns = [
            DialogueTurn(speaker=turn.role, text=turn.text) 
            for turn in generated_dialogue.dialogue
        ]

        # 3. LLM Analysis directly on the dialogue
        logger.info("Starting LLM analysis on generated dialogue...")
        system_prompt: str = get_analysis_prompt()
        analysis: AnalysisResult = await self.llm.analyze(transcript_turns, system_prompt)
        logger.info("LLM analysis completed successfully.")

        return transcript_turns, analysis


# Singleton or Factory
def get_session_service() -> MedicalSessionService:
    return MedicalSessionService()
