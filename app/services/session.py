import asyncio
import uuid

from app.core.models import AnalysisResult, DialogueTurn, GeneratedDialogue, ImageAttachment
from app.services.llm import get_llm_provider
from app.services.stt import get_stt_provider
from app.services.tts import get_tts_provider
from config.logger import logger
from config.prompts import get_analysis_prompt, get_dialogue_generation_prompt
from config.settings import config


class MedicalSessionService:
    def __init__(self):
        self._stt = get_stt_provider()
        self._llm = get_llm_provider()
        self._tts = get_tts_provider()
        logger.info(f"MedicalSessionService initialized. Use Mock: {config.USE_MOCK_SERVICES}")

    async def process_upload(
        self, audio_path: str, images: list[ImageAttachment] | None = None
    ) -> tuple[list[DialogueTurn], str | None]:
        """Processes uploaded audio and images in parallel."""
        logger.info(f"Processing upload: audio={audio_path}, images={len(images) if images else 0}")

        stt_task = self._stt.transcribe(audio_path)
        image_task = self._llm.analyze_images(images) if images else asyncio.sleep(0)

        results = await asyncio.gather(stt_task, image_task)

        transcript = results[0]
        image_report = results[1] if images else None

        return transcript, image_report

    async def generate_simulation(
        self, diagnosis: str | None, doctor_skill: int, images: list[ImageAttachment] | None = None
    ) -> tuple[list[DialogueTurn], str | None]:
        """Generates a simulated dialogue and processes images in parallel."""
        logger.info(
            f"Generating simulation: diagnosis={diagnosis}, skill={doctor_skill}, images={len(images) if images else 0}"
        )

        system_prompt = get_dialogue_generation_prompt(
            diagnosis=diagnosis, doctor_skill=doctor_skill
        )

        gen_task = self._llm.generate_dialogue(system_prompt=system_prompt, diagnosis=diagnosis)
        image_task = self._llm.analyze_images(images) if images else asyncio.sleep(0)

        results = await asyncio.gather(gen_task, image_task)

        generated_dialogue: GeneratedDialogue = results[0]
        image_report = results[1] if images else None

        transcript_turns = [
            DialogueTurn(speaker=turn.role, text=turn.text) for turn in generated_dialogue.dialogue
        ]

        return transcript_turns, image_report

    async def analyze_consultation(
        self, transcript: list[DialogueTurn], image_report: str | None = None
    ) -> AnalysisResult:
        """Analyzes the consultation transcript and image report."""
        logger.info("Starting final analysis...")
        system_prompt = get_analysis_prompt()
        analysis = await self._llm.analyze(transcript, system_prompt, image_report)
        logger.info("Final analysis completed.")
        return analysis

    async def generate_voice_recommendations(self, recommendations: list[str]) -> str:
        logger.info(f"Generating voice recommendations for {len(recommendations)} items.")
        text_to_speak = "Рекомендации по назначению: " + ". ".join(recommendations)
        filename = f"recs_{uuid.uuid4()}.mp3"
        output_path = config.TEMP_DIR / filename

        audio_file = await self._tts.speak(text_to_speak, str(output_path))
        logger.info(f"Audio generated at: {audio_file}")
        return audio_file

    async def text_to_speech(self, text: str, voice: str | None = None) -> str:
        logger.info(f"Generating speech for text (length: {len(text)})...")
        filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
        output_path = config.TEMP_DIR / filename

        audio_file = await self._tts.speak(text=text, output_path=str(output_path), voice=voice)
        logger.info(f"Audio generated at: {audio_file}")
        return audio_file


# Singleton or Factory
def get_session_service() -> MedicalSessionService:
    return MedicalSessionService()
