import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from app.core.interfaces import TTSProvider
from config.logger import logger
from config.settings import config


class MockTTS(TTSProvider):
    async def speak(self, text: str, output_path: str, voice: str | None = None) -> str:
        logger.info(f"MockTTS: Generating speech for text (len={len(text)}), voice={voice}")
        await asyncio.sleep(1)
        # In a real mock, we might copy a pre-recorded file here
        # For now, we assume the file exists or we create a dummy one
        # This is just a placeholder implementation
        if not Path(output_path).exists():
            with open(output_path, "wb") as f:
                f.write(b"dummy audio content")
        return output_path


class OpenAITTS(TTSProvider):
    instructions = """Voice: Professional, direct, and quick.
    Tone: Confident and concise, suitable for delivering medical instructions or recommendations.
    Speech Mannerisms: Uses clear, straightforward, and formal phrasing with no slang, focusing on clarity and efficiency.
    Pronunciation: Precise and articulate, with crisp consonants and accurate medical terminology.
    Tempo: Fast and purposeful, moving efficiently through the information while maintaining clarity.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def speak(self, text: str, output_path: str, voice: str | None = None) -> str:
        target_voice = voice or "alloy"
        logger.info(f"OpenAITTS: Synthesizing speech to {output_path} with voice {target_voice}")
        async with self.client.audio.speech.with_streaming_response.create(
            model=config.TTS_MODEL, voice=target_voice, input=text, instructions=self.instructions
        ) as response:
            await response.stream_to_file(output_path)
        return output_path


def get_tts_provider() -> TTSProvider:
    if config.USE_MOCK_SERVICES:
        return MockTTS()
    return OpenAITTS()
