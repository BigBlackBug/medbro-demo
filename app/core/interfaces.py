from abc import ABC, abstractmethod

from app.core.models import AnalysisResult


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str) -> str:
        """Transcribes audio file to text."""


class LLMProvider(ABC):
    @abstractmethod
    async def analyze(self, text: str, system_prompt: str) -> AnalysisResult:
        """Analyzes text and returns structured result."""


class TTSProvider(ABC):
    @abstractmethod
    async def speak(self, text: str, output_path: str) -> str:
        """Converts text to speech and saves to output_path. Returns path."""
