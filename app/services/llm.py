import asyncio

from openai import AsyncOpenAI

from app.core.interfaces import LLMProvider
from app.core.models import (
    AnalysisResult,
    DoctorEvaluation,
    EvaluationCriterion,
    Medication,
    PrescriptionReview,
    StructuredData,
)
from config.logger import logger
from config.settings import config


class MockLLM(LLMProvider):
    async def analyze(self, text: str, system_prompt: str) -> AnalysisResult:
        logger.info("MockLLM: Analyzing text...")
        await asyncio.sleep(2)
        return AnalysisResult(
            structured_data=StructuredData(
                complaints=["сильный кашель", "температура 38", "3 день"],
                diagnosis="ОРВИ (предварительно)",
                medications=[
                    Medication(
                        name="Амоксиклав",
                        dosage="875 мг",
                        frequency="2 раза в день",
                        duration="7 дней",
                    )
                ],
            ),
            prescription_review=PrescriptionReview(
                status="warning",
                recommendations=[
                    "Не уточнено наличие аллергии на пенициллины (пациент не уверен).",
                    "Антибиотик назначен эмпирически без анализа крови (возможно, вирусная этиология).",
                    "Не назначены пробиотики.",
                ],
            ),
            doctor_evaluation=DoctorEvaluation(
                criteria={
                    "history_taking": EvaluationCriterion(
                        score=4, comment="Собрал основные жалобы, но не дожал тему аллергии."
                    ),
                    "clinical_reasoning": EvaluationCriterion(
                        score=3,
                        comment="Назначил антибиотик сразу, без подтверждения бак. инфекции.",
                    ),
                    "communication": EvaluationCriterion(score=5, comment="Вежлив, понятен."),
                    "safety": EvaluationCriterion(score=3, comment="Риск аллергической реакции."),
                },
                general_comment="Врач действовал по стандартному протоколу, но стоит быть внимательнее к аллергоанамнезу.",
            ),
        )


class OpenAILLM(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def analyze(self, text: str, system_prompt: str) -> AnalysisResult:
        logger.info(f"OpenAILLM: Sending request to {config.LLM_MODEL}")
        response = await self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            logger.error("OpenAILLM: Received empty response")
            raise ValueError("Empty response from LLM")

        logger.info("OpenAILLM: Received valid response")
        return AnalysisResult.model_validate_json(content)


def get_llm_provider() -> LLMProvider:
    if config.USE_MOCK_SERVICES:
        return MockLLM()
    return OpenAILLM()
