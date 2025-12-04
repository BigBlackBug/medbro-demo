from abc import ABC, abstractmethod

from app.core.models import AnalysisResult, DialogueTurn, GeneratedDialogue, ImageAttachment


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str) -> list[DialogueTurn]:
        """Transcribes audio file to text with diarization."""

    @abstractmethod
    async def transcribe_raw(self, audio_path: str) -> str:
        """Transcribes audio file to raw text without diarization."""


class LLMProvider(ABC):
    @abstractmethod
    async def analyze(self, dialogue: list[DialogueTurn], system_prompt: str, images: list[ImageAttachment] | None = None) -> AnalysisResult:
        """Analyzes dialogue and returns structured result."""

    @abstractmethod
    async def analyze_raw(self, text: str, system_prompt: str, images: list[ImageAttachment] | None = None) -> AnalysisResult:
        """Analyzes raw text and returns structured result."""

    @abstractmethod
    async def generate_dialogue(self, system_prompt: str, diagnosis: str | None = None) -> GeneratedDialogue:
        """Generates a realistic doctor-patient dialogue."""


class TTSProvider(ABC):
    @abstractmethod
    async def speak(self, text: str, output_path: str, voice: str | None = None) -> str:
        """Converts text to speech and saves to output_path. Returns path.

        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            voice: Optional voice name (e.g. "alloy", "sage", "fable"). If None, uses default.
        """
