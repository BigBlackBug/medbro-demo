import asyncio

from openai import AsyncOpenAI

from app.core.interfaces import STTProvider
from app.core.models import DialogueTurn
from config.logger import logger
from config.settings import config


class MockSTT(STTProvider):
    async def transcribe(self, audio_path: str) -> list[DialogueTurn]:
        logger.info(f"MockSTT: Transcribing {audio_path}")
        await asyncio.sleep(2)  # Simulate processing
        return [
            DialogueTurn(speaker="Врач", text="Добрый день. На что жалуетесь?"),
            DialogueTurn(
                speaker="Пациент",
                text="Здравствуйте, доктор. У меня сильный кашель и температура 38 уже третий день.",
            ),
            DialogueTurn(speaker="Врач", text="Понятно. Есть ли мокрота? Аллергия на лекарства?"),
            DialogueTurn(
                speaker="Пациент", text="Мокроты нет, кашель сухой. Аллергии вроде нет, но я не уверен."
            ),
            DialogueTurn(
                speaker="Врач",
                text="Хорошо. Я назначу вам Амоксиклав 875 мг два раза в день на 7 дней. Пейте больше жидкости.",
            ),
            DialogueTurn(speaker="Пациент", text="Спасибо, доктор."),
        ]

    async def transcribe_raw(self, audio_path: str) -> str:
        logger.info(f"MockSTT: Transcribing raw {audio_path}")
        await asyncio.sleep(1)
        return (
            "Врач: Добрый день. На что жалуетесь?\n"
            "Пациент: Здравствуйте, доктор. У меня сильный кашель и температура 38 уже третий день.\n"
            "Врач: Понятно. Есть ли мокрота? Аллергия на лекарства?\n"
            "Пациент: Мокроты нет, кашель сухой. Аллергии вроде нет, но я не уверен.\n"
            "Врач: Хорошо. Я назначу вам Амоксиклав 875 мг два раза в день на 7 дней. "
            "Пейте больше жидкости.\n"
            "Пациент: Спасибо, доктор."
        )


class OpenAI_STT(STTProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def transcribe(self, audio_path: str) -> list[DialogueTurn]:
        # Use the specific model requested for diarization
        
        logger.info(f"OpenAI_STT: Transcribing {audio_path} with model {config.STT_DIARIZATION_MODEL}")
        
        with open(audio_path, "rb") as audio_file:
            # Based on user documentation snippet
            transcript = await self.client.audio.transcriptions.create(
                model=config.STT_DIARIZATION_MODEL,
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",
            )
            
        dialogue = []
        for segment in transcript.segments:
            speaker = getattr(segment, "speaker", "Unknown")
            text = segment.text.strip()
            if text:
                logger.info(f"OpenAI_STT: Transcribed {text} from {speaker}")
                dialogue.append(DialogueTurn(speaker=speaker, text=text))
        
        return dialogue

    async def transcribe_raw(self, audio_path: str) -> str:
        logger.info(f"OpenAI_STT: Transcribing raw {audio_path} with model {config.STT_MODEL}")
        with open(audio_path, "rb") as audio_file:
            transcript = await self.client.audio.transcriptions.create(
                model=config.STT_MODEL, file=audio_file
            )
        return transcript.text


def get_stt_provider() -> STTProvider:
    if config.USE_MOCK_SERVICES:
        return MockSTT()
    return OpenAI_STT()
