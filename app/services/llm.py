import asyncio
import base64
from pathlib import Path

from openai import AsyncOpenAI

from app.core.interfaces import LLMProvider
from app.core.models import (
    AnalysisResult,
    DialogueTurn,
    DoctorEvaluation,
    EvaluationCriterion,
    GeneratedDialogue,
    GeneratedDialogueTurn,
    ImageAttachment,
    Medication,
    PrescriptionReview,
    StructuredData,
)
from config.logger import logger
from config.settings import config


class MockLLM(LLMProvider):
    async def analyze(self, dialogue: list[DialogueTurn], system_prompt: str, images: list[ImageAttachment] | None = None) -> AnalysisResult:
        logger.info("MockLLM: Analyzing dialogue...")
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
                criteria=[
                    EvaluationCriterion(
                        name="history_taking",
                        score=4,
                        comment="Собрал основные жалобы, но не дожал тему аллергии.",
                    ),
                    EvaluationCriterion(
                        name="clinical_reasoning",
                        score=3,
                        comment="Назначил антибиотик сразу, без подтверждения бак. инфекции.",
                    ),
                    EvaluationCriterion(
                        name="communication",
                        score=5,
                        comment="Вежлив, понятен.",
                    ),
                    EvaluationCriterion(
                        name="safety",
                        score=3,
                        comment="Риск аллергической реакции.",
                    ),
                ],
                general_comment="Врач действовал по стандартному протоколу, но стоит быть внимательнее к аллергоанамнезу.",
            ),
            formatted_transcript=(
                "Доктор: Добрый день. На что FFFF?<br>"
                "Пациент: У меня <span style='background-color: #ffeef0; color: #b31b1b;'>сильный кашель</span> и "
                "<span style='background-color: #ffeef0; color: #b31b1b;'>температура 38</span> уже "
                "<span style='background-color: #ffeef0; color: #b31b1b;'>3 день</span>.<br>"
                "Доктор: Понятно. Аллергии есть?<br>"
                "Пациент: Не знаю, вроде нет.<br>"
                "Доктор: Хорошо. Принимайте <span style='background-color: #e6ffed; color: #22863a;'>Амоксиклав 875 мг 2 раза в день</span> "
                "в течение <span style='background-color: #e6ffed; color: #22863a;'>7 дней</span>."
            ),
        )

    async def analyze_raw(self, text: str, system_prompt: str, images: list[ImageAttachment] | None = None) -> AnalysisResult:
        logger.info("MockLLM: Analyzing raw text...")
        return await self.analyze([], system_prompt, images)

    async def generate_dialogue(self, system_prompt: str, diagnosis: str | None = None) -> GeneratedDialogue:
        logger.info(f"MockLLM: Generating dialogue{f' for diagnosis: {diagnosis}' if diagnosis else ''}...")
        await asyncio.sleep(1)
        return GeneratedDialogue(
            dialogue=[
                GeneratedDialogueTurn(
                    role="Врач", voice="sage", text="Добрый день. На что жалуетесь?"
                ),
                GeneratedDialogueTurn(
                    role="Пациент",
                    voice="fable",
                    text="Здравствуйте, голова болит уже третий день.",
                ),
                GeneratedDialogueTurn(
                    role="Врач", voice="sage", text="Как болит? Пульсирует или давит?"
                ),
                GeneratedDialogueTurn(
                    role="Пациент", voice="fable", text="Давит, как обручем стянуло."
                ),
                GeneratedDialogueTurn(role="Врач", voice="sage", text="Понятно. Давление мерили?"),
                GeneratedDialogueTurn(role="Пациент", voice="fable", text="Нет, не мерил."),
            ]
        )


class OpenAILLM(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_image_mime_type(self, image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(suffix, "image/jpeg")

    def _build_messages_with_images(self, text_content: str, system_prompt: str, images: list[ImageAttachment] | None = None) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        
        if not images:
            messages.append({"role": "user", "content": text_content})
        else:
            content_parts = [{"type": "text", "text": text_content}]
            
            for img in images:
                try:
                    base64_image = self._encode_image(img.file_path)
                    mime_type = self._get_image_mime_type(img.file_path)
                    
                    image_description = ""
                    if img.description:
                        image_description = f"\n\nImage description: {img.description}"
                    if img.image_type:
                        image_description += f"\nImage type: {img.image_type}"
                    
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                    
                    if image_description:
                        content_parts.append({"type": "text", "text": image_description})
                        
                    logger.info(f"Added image to request: {img.file_path}")
                except Exception as e:
                    logger.error(f"Failed to encode image {img.file_path}: {e}")
            
            messages.append({"role": "user", "content": content_parts})
        
        return messages

    async def analyze(self, dialogue: list[DialogueTurn], system_prompt: str, images: list[ImageAttachment] | None = None) -> AnalysisResult:
        dialogue_text = "\n".join([f"{turn.speaker}: {turn.text}" for turn in dialogue])
        
        if images:
            logger.info(f"OpenAILLM: Sending request to {config.LLM_MODEL} with {len(images)} image(s)")
            dialogue_text = f"Consultation dialogue:\n{dialogue_text}\n\nPlease analyze the dialogue along with the provided medical images."
        else:
            logger.info(f"OpenAILLM: Sending request to {config.LLM_MODEL}")
        
        messages = self._build_messages_with_images(dialogue_text, system_prompt, images)
        
        response = await self.client.beta.chat.completions.parse(
            model=config.LLM_MODEL,
            messages=messages,
            response_format=AnalysisResult,
        )

        parsed_result = response.choices[0].message.parsed
        if not parsed_result:
            logger.error("OpenAILLM: Received empty or invalid response")
            raise ValueError("Empty or invalid response from LLM")

        logger.info("OpenAILLM: Received valid response")
        return parsed_result

    async def analyze_raw(self, text: str, system_prompt: str, images: list[ImageAttachment] | None = None) -> AnalysisResult:
        if images:
            logger.info(f"OpenAILLM: Sending raw request to {config.LLM_MODEL} with {len(images)} image(s)")
            text = f"{text}\n\nPlease analyze the text along with the provided medical images."
        else:
            logger.info(f"OpenAILLM: Sending raw request to {config.LLM_MODEL}")
        
        messages = self._build_messages_with_images(text, system_prompt, images)
        
        response = await self.client.beta.chat.completions.parse(
            model=config.LLM_MODEL,
            messages=messages,
            response_format=AnalysisResult,
        )

        parsed_result = response.choices[0].message.parsed
        if not parsed_result:
            logger.error("OpenAILLM: Received empty or invalid response")
            raise ValueError("Empty or invalid response from LLM")

        logger.info("OpenAILLM: Received valid response")
        return parsed_result

    async def generate_dialogue(self, system_prompt: str, diagnosis: str | None = None) -> GeneratedDialogue:
        logger.info(f"OpenAILLM: Generating dialogue with {config.LLM_MODEL}{f' for diagnosis: {diagnosis}' if diagnosis else ''}")
        response = await self.client.beta.chat.completions.parse(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate a realistic doctor-patient dialogue."},
            ],
            response_format=GeneratedDialogue,
        )

        parsed_result = response.choices[0].message.parsed
        if not parsed_result:
            logger.error("OpenAILLM: Received empty or invalid response for dialogue generation")
            raise ValueError("Empty or invalid response from LLM")

        logger.info("OpenAILLM: Received generated dialogue")
        return parsed_result


def get_llm_provider() -> LLMProvider:
    if config.USE_MOCK_SERVICES:
        return MockLLM()
    return OpenAILLM()
