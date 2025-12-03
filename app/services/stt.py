import asyncio

from openai import AsyncOpenAI

from app.core.interfaces import STTProvider
from config.logger import logger
from config.settings import config


class MockSTT(STTProvider):
    async def transcribe(self, audio_path: str) -> str:
        logger.info(f"MockSTT: Transcribing {audio_path}")
        await asyncio.sleep(2)  # Simulate processing
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

    async def transcribe(self, audio_path: str) -> str:
        logger.info(f"OpenAI_STT: Transcribing {audio_path} with model {config.STT_MODEL}")
        with open(audio_path, "rb") as audio_file:
            transcript = await self.client.audio.transcriptions.create(
                model=config.STT_MODEL, file=audio_file
            )
        return transcript.text


def get_stt_provider() -> STTProvider:
    if config.USE_MOCK_SERVICES:
        return MockSTT()
    return OpenAI_STT()
