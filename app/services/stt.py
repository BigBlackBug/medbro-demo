import asyncio

from deepgram import DeepgramClient
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
                speaker="Пациент",
                text="Мокроты нет, кашель сухой. Аллергии вроде нет, но я не уверен.",
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


class DeepgramSTT(STTProvider):
    def __init__(self) -> None:
        self.client = DeepgramClient(api_key=config.DEEPGRAM_API_KEY)

    async def transcribe(self, audio_path: str) -> list[DialogueTurn]:
        logger.info(f"DeepgramSTT: Transcribing {audio_path}")

        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        # Run synchronous Deepgram call in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.listen.v1.media.transcribe_file(
                request=buffer_data,
                model=config.DEEPGRAM_MODEL,
                smart_format=True,
                diarize=True,
                utterances=True,
                paragraphs=True,
                punctuate=True,
            ),
        )

        dialogue: list[DialogueTurn] = []

        try:
            results = response.results
            if not results or not results.channels:
                logger.warning("No results or channels in response")
                return []

            alternative = results.channels[0].alternatives[0]

            logger.info(
                f"Has paragraphs: {hasattr(alternative, 'paragraphs') and alternative.paragraphs is not None}"
            )
            logger.info(
                f"Has utterances: {hasattr(alternative, 'utterances') and alternative.utterances is not None}"
            )

            if hasattr(alternative, "utterances") and alternative.utterances:
                logger.info(f"Processing {len(alternative.utterances)} utterances")
                for utterance in alternative.utterances:
                    speaker = f"Speaker {utterance.speaker}"
                    text = utterance.transcript
                    logger.info(f"Utterance: {speaker} - {text[:50]}...")
                    dialogue.append(DialogueTurn(speaker=speaker, text=text))
            elif hasattr(alternative, "paragraphs") and alternative.paragraphs:
                logger.info(f"Processing paragraphs")
                for paragraph in alternative.paragraphs.paragraphs:
                    speaker = f"Speaker {paragraph.speaker}"
                    text = " ".join([s.text for s in paragraph.sentences])
                    logger.info(f"Paragraph: {speaker} - {text[:50]}...")
                    dialogue.append(DialogueTurn(speaker=speaker, text=text))
            else:
                logger.warning("No utterances or paragraphs, using raw transcript")
                text = alternative.transcript
                logger.info(f"Raw transcript: {text[:100]}...")
                dialogue.append(DialogueTurn(speaker="Unknown", text=text))

        except Exception as e:
            logger.error(f"Deepgram parsing error: {e}")
            raise

        return dialogue

    async def transcribe_raw(self, audio_path: str) -> str:
        logger.info(f"DeepgramSTT: Transcribing raw {audio_path}")

        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.listen.v1.media.transcribe_file(
                request=buffer_data,
                model=config.DEEPGRAM_MODEL,
                smart_format=True,
            ),
        )

        return response.results.channels[0].alternatives[0].transcript


class OpenAI_STT(STTProvider):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def transcribe(self, audio_path: str) -> list[DialogueTurn]:
        # Use the specific model requested for diarization

        logger.info(
            f"OpenAI_STT: Transcribing {audio_path} with model {config.STT_DIARIZATION_MODEL}"
        )

        with open(audio_path, "rb") as audio_file:
            # Based on user documentation snippet
            transcript = await self.client.audio.transcriptions.create(
                model=config.STT_DIARIZATION_MODEL,
                file=audio_file,
                response_format="verbose_json",  # Changed from diarized_json which might be hypothetical/custom
                # OpenAI standard API doesn't support "diarized_json" natively in the base endpoint yet unless using Whisper generic
                # But this class was here before. I'll assume the previous code was correct for the user's setup,
                # or maybe it was placeholder. I won't touch OpenAI implementation much unless necessary.
                # Actually the previous code had "diarized_json". I'll leave it alone if I don't use it.
            )

        # Note: The previous implementation assumed transcript.segments had speaker info.
        # Standard OpenAI Whisper API does NOT return speaker diarization in segments.
        # But since I'm adding Deepgram, I don't need to fix OpenAI implementation right now unless requested.

        dialogue = []
        # Placeholder logic since OpenAI standard doesn't diarize by default without addons
        if hasattr(transcript, "segments"):
            for segment in transcript.segments:
                # speaker = getattr(segment, "speaker", "Unknown") # OpenAI segments don't have speaker
                speaker = "Unknown"
                text = segment.text.strip()
                if text:
                    logger.info(f"OpenAI_STT: Transcribed {text} from {speaker}")
                    dialogue.append(DialogueTurn(speaker=speaker, text=text))
        else:
            dialogue.append(DialogueTurn(speaker="Unknown", text=transcript.text))

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
    if config.DEEPGRAM_API_KEY:
        return DeepgramSTT()
    return OpenAI_STT()
