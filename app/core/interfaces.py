from abc import ABC, abstractmethod

from app.core.models import AnalysisResult, DialogueTurn


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str) -> list[DialogueTurn]:
        """Transcribes audio file to text with diarization."""

    @abstractmethod
    async def transcribe_raw(self, audio_path: str) -> str:
        """Transcribes audio file to raw text without diarization."""


class LLMProvider(ABC):
    @abstractmethod
    async def analyze(self, dialogue: list[DialogueTurn], system_prompt: str) -> AnalysisResult:
        """Analyzes dialogue and returns structured result."""

    @abstractmethod
    async def analyze_raw(self, text: str, system_prompt: str) -> AnalysisResult:
        """Analyzes raw text and returns structured result."""


class TTSProvider(ABC):
    @abstractmethod
    async def speak(self, text: str, output_path: str) -> str:
        """Converts text to speech and saves to output_path. Returns path."""
