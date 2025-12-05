import asyncio
import re
import uuid
from pathlib import Path

from app.core.models import AnalysisResult, DialogueTurn, GeneratedDialogue, ImageAttachment
from app.services.llm import get_llm_provider
from app.services.stt import get_stt_provider
from app.services.tts import get_tts_provider
from config.logger import logger
from config.prompts import get_analysis_prompt, get_dialogue_generation_prompt
from config.settings import config


class MedicalSessionService:
    def __init__(self) -> None:
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

    async def generate_dialogue_audio(
        self, diagnosis: str | None, doctor_skill: int, output_dir: Path
    ) -> str:
        logger.info(f"Generating dialogue audio: diagnosis={diagnosis}, skill={doctor_skill}")

        output_dir.mkdir(exist_ok=True)

        diagnosis = diagnosis.strip() if diagnosis else None
        if not diagnosis:
            diagnosis = None

        file_index = self._get_next_file_index(output_dir)
        diagnosis_slug: str = (
            self._sanitize_diagnosis_for_filename(diagnosis) if diagnosis else "random"
        )
        output_file = (
            output_dir / f"{file_index:04d}_dialogue_{diagnosis_slug}_skill{doctor_skill}.mp3"
        )

        system_prompt = get_dialogue_generation_prompt(
            diagnosis=diagnosis, doctor_skill=doctor_skill
        )
        generated_dialogue = await self._llm.generate_dialogue(
            system_prompt=system_prompt, diagnosis=diagnosis
        )
        dialogue_script = generated_dialogue.dialogue

        logger.info(f"Dialogue generated with {len(dialogue_script)} turns")

        temp_files: list[Path] = []

        try:
            for idx, turn in enumerate(dialogue_script):
                role = turn.role
                voice = turn.voice
                text = turn.text

                logger.info(f"[{idx+1}/{len(dialogue_script)}] {role}: {text[:30]}...")

                temp_file = output_dir / f"part_{idx}_{role}_{uuid.uuid4().hex[:8]}.mp3"
                temp_files.append(temp_file)

                await self._tts.speak(text=text, output_path=str(temp_file), voice=voice)

            logger.info("Combining audio segments...")

            with open(output_file, "wb") as outfile:
                for fpath in temp_files:
                    with open(fpath, "rb") as infile:
                        outfile.write(infile.read())

            logger.info(f"Audio generation complete: {output_file}")
            return str(output_file)

        finally:
            logger.info("Cleaning up temporary files...")
            for fpath in temp_files:
                if fpath.exists():
                    fpath.unlink()

    def _sanitize_diagnosis_for_filename(self, diagnosis: str) -> str:
        slug = diagnosis.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "_", slug)
        slug = slug.strip("_")
        return slug

    def _get_next_file_index(self, output_dir: Path) -> int:
        existing_files = list(output_dir.glob("*_dialogue_*.mp3"))
        if not existing_files:
            return 1

        max_index = 0
        for file in existing_files:
            try:
                index = int(file.name.split("_")[0])
                max_index = max(max_index, index)
            except (ValueError, IndexError):
                continue

        return max_index + 1


# Singleton or Factory
def get_session_service() -> MedicalSessionService:
    return MedicalSessionService()
