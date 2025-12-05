from collections.abc import AsyncIterator
from typing import Any

from app.core.models import (
    AnalysisResult,
    ComplaintsResponse,
    DiagnosisResponse,
    DialogueTurn,
    DoctorEvaluation,
    EvaluationCriterion,
    GeneratedDialogue,
    ImageAttachment,
    Medication,
    MedicationsResponse,
    PrescriptionReview,
    StructuredData,
)
from app.services.llm import OpenAILLM
from app.services.stt import get_stt_provider
from config.logger import logger
from config.prompts import (
    get_complaints_streaming_prompt,
    get_criteria_streaming_prompt,
    get_diagnosis_streaming_prompt,
    get_dialogue_generation_prompt,
    get_general_comment_streaming_prompt,
    get_medications_streaming_prompt,
    get_recommendations_streaming_prompt,
    get_transcript_streaming_prompt,
)


class MedicalSessionStreamingService:
    def __init__(self) -> None:
        self._stt = get_stt_provider()
        self._llm = OpenAILLM()
        logger.info("MedicalSessionStreamingService initialized")

    def _parse_criterion(self, text: str) -> EvaluationCriterion | None:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        criterion_name: str | None = None
        score: int | None = None
        comment: str | None = None

        for line in lines:
            if line.startswith("CRITERION_NAME:"):
                criterion_name = line.replace("CRITERION_NAME:", "").strip()
            elif line.startswith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:", "").strip())
                except ValueError:
                    logger.warning(f"Failed to parse score: {line}")
            elif line.startswith("COMMENT:"):
                comment = line.replace("COMMENT:", "").strip()

        if criterion_name and score is not None and comment:
            return EvaluationCriterion(name=criterion_name, score=score, comment=comment)

        logger.warning(f"Failed to parse criterion from text: {text}")
        return None

    async def process_upload(
        self, audio_path: str, images: list[ImageAttachment] | None = None
    ) -> tuple[list[DialogueTurn], str | None]:
        logger.info(f"Processing upload: audio={audio_path}, images={len(images) if images else 0}")

        image_report: str | None = None
        if images:
            logger.info("→ Starting image analysis first...")
            image_report = await self._llm.analyze_images(images)
            logger.info(f"→ Image analysis complete: {len(image_report)} chars")

        logger.info("→ Starting transcription...")
        transcript = await self._stt.transcribe(audio_path)
        logger.info(f"→ Transcription complete: {len(transcript)} turns")

        return transcript, image_report

    async def generate_simulation(
        self,
        diagnosis: str | None,
        doctor_skill: int,
        images: list[ImageAttachment] | None = None,
    ) -> tuple[list[DialogueTurn], str | None]:
        logger.info(
            f"Generating simulation: diagnosis={diagnosis}, skill={doctor_skill}, images={len(images) if images else 0}"
        )

        image_report: str | None = None
        if images:
            logger.info("→ Starting image analysis first...")
            image_report = await self._llm.analyze_images(images)
            logger.info(f"→ Image analysis complete: {len(image_report)} chars")

        system_prompt = get_dialogue_generation_prompt(
            diagnosis=diagnosis, doctor_skill=doctor_skill
        )

        logger.info("→ Starting dialogue generation...")
        generated_dialogue: GeneratedDialogue = await self._llm.generate_dialogue(
            system_prompt=system_prompt, diagnosis=diagnosis
        )
        logger.info(f"→ Dialogue generation complete: {len(generated_dialogue.dialogue)} turns")

        transcript_turns = [
            DialogueTurn(speaker=turn.role, text=turn.text) for turn in generated_dialogue.dialogue
        ]

        return transcript_turns, image_report

    async def analyze_consultation_streaming(
        self, transcript: list[DialogueTurn], image_report: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        logger.info("Starting streaming analysis...")

        response_id: str | None = None
        formatted_transcript = ""
        complaints: list[str] = []
        diagnosis: str | None = None
        medications: list[Medication] = []
        recommendations: list[str] = []
        criteria: list[EvaluationCriterion] = []
        general_comment = ""

        try:
            logger.info("→ Starting transcript stage")
            yield {"stage": "transcript", "status": "starting", "data": None}

            stream_manager = await self._llm.analyze_formatted_transcript_streaming(
                dialogue=transcript,
                system_prompt=get_transcript_streaming_prompt(),
            )

            formatted_transcript = ""
            async with stream_manager as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        if hasattr(event, "delta") and event.delta:
                            formatted_transcript += event.delta
                            # logger.info(
                            #     f"→ Streaming transcript delta (buffer: {len(formatted_transcript)} chars)"
                            # )
                            yield {
                                "stage": "transcript",
                                "status": "streaming",
                                "data": formatted_transcript,
                            }
                    elif event.type == "response.completed":
                        logger.info("→ Transcript stream completed")

                final_response = await stream.get_final_response()
                response_id = final_response.id

            logger.info(f"→ Transcript complete. Response ID: {response_id}")
            yield {"stage": "transcript", "status": "complete", "data": formatted_transcript}

            logger.info("→ Starting complaints stage")
            yield {"stage": "complaints", "status": "starting", "data": None}

            stream_manager = await self._llm.analyze_complaints_streaming(
                previous_response_id=response_id,
                system_prompt=get_complaints_streaming_prompt(),
            )

            async with stream_manager as stream:
                async for event in stream:
                    if event.type == "response.completed":
                        logger.info("→ Complaints stream completed")

                final_response = await stream.get_final_response()
                response_id = final_response.id

                if final_response.output:
                    for message in final_response.output:
                        if hasattr(message, "content"):
                            for content in message.content:
                                if hasattr(content, "parsed"):
                                    parsed_complaints: ComplaintsResponse = content.parsed
                                    complaints = parsed_complaints.complaints

            logger.info(f"→ Complaints complete: {len(complaints)} items")
            yield {"stage": "complaints", "status": "streaming", "data": complaints}
            yield {"stage": "complaints", "status": "complete", "data": complaints}

            logger.info("→ Starting diagnosis stage")
            yield {"stage": "diagnosis", "status": "starting", "data": None}

            stream_manager = await self._llm.analyze_diagnosis_streaming(
                previous_response_id=response_id,
                system_prompt=get_diagnosis_streaming_prompt(),
                image_analysis=image_report,
            )

            async with stream_manager as stream:
                async for event in stream:
                    if event.type == "response.completed":
                        logger.info("→ Diagnosis stream completed")

                final_response = await stream.get_final_response()
                response_id = final_response.id

                if final_response.output:
                    for message in final_response.output:
                        if hasattr(message, "content"):
                            for content in message.content:
                                if hasattr(content, "parsed"):
                                    parsed_diagnosis: DiagnosisResponse = content.parsed
                                    diagnosis = parsed_diagnosis.diagnosis

            logger.info(f"→ Diagnosis complete: {diagnosis}")
            yield {"stage": "diagnosis", "status": "streaming", "data": diagnosis}
            yield {"stage": "diagnosis", "status": "complete", "data": diagnosis}

            logger.info("→ Starting medications stage")
            yield {"stage": "medications", "status": "starting", "data": None}

            stream_manager = await self._llm.analyze_medications_streaming(
                previous_response_id=response_id,
                system_prompt=get_medications_streaming_prompt(),
            )

            async with stream_manager as stream:
                async for event in stream:
                    if event.type == "response.completed":
                        logger.info("→ Medications stream completed")

                final_response = await stream.get_final_response()
                response_id = final_response.id

                if final_response.output:
                    for message in final_response.output:
                        if hasattr(message, "content"):
                            for content in message.content:
                                if hasattr(content, "parsed"):
                                    parsed_medications: MedicationsResponse = content.parsed
                                    medications = parsed_medications.medications

            logger.info(f"→ Medications complete: {len(medications)} medications")
            yield {"stage": "medications", "status": "streaming", "data": medications}
            yield {"stage": "medications", "status": "complete", "data": medications}

            logger.info("→ Starting recommendations stage")
            yield {"stage": "recommendations", "status": "starting", "data": None}

            stream_manager = await self._llm.analyze_recommendations_streaming(
                previous_response_id=response_id,
                system_prompt=get_recommendations_streaming_prompt(),
                image_analysis=image_report,
            )

            buffer = ""
            async with stream_manager as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        if hasattr(event, "delta") and event.delta:
                            buffer += event.delta

                            if "__ITEM__" in buffer:
                                parts = buffer.split("__ITEM__")
                                for i in range(len(parts) - 1):
                                    item_text = parts[i].strip()
                                    if item_text and item_text.lower() != "no recommendations.":
                                        recommendations.append(item_text)
                                        yield {
                                            "stage": "recommendations",
                                            "status": "streaming",
                                            "data": recommendations,
                                        }
                                buffer = parts[-1]
                    elif event.type == "response.completed":
                        logger.info("→ Recommendations stream completed")

                final_response = await stream.get_final_response()
                response_id = final_response.id

            if buffer.strip() and buffer.strip().lower() != "no recommendations.":
                recommendations.append(buffer.strip())

            logger.info(f"→ Recommendations complete: {len(recommendations)} recommendations")
            yield {"stage": "recommendations", "status": "complete", "data": recommendations}

            logger.info("→ Starting criteria stage")
            yield {"stage": "criteria", "status": "starting", "data": None}

            stream_manager = await self._llm.analyze_criteria_streaming(
                previous_response_id=response_id,
                system_prompt=get_criteria_streaming_prompt(),
            )

            buffer = ""
            async with stream_manager as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        if hasattr(event, "delta") and event.delta:
                            buffer += event.delta

                            if "__ITEM__" in buffer:
                                parts = buffer.split("__ITEM__")
                                for i in range(len(parts) - 1):
                                    item_text = parts[i].strip()
                                    if item_text:
                                        criterion = self._parse_criterion(item_text)
                                        if criterion:
                                            criteria.append(criterion)
                                            yield {
                                                "stage": "criteria",
                                                "status": "streaming",
                                                "data": criteria,
                                            }
                                buffer = parts[-1]
                    elif event.type == "response.completed":
                        logger.info("→ Criteria stream completed")

                final_response = await stream.get_final_response()
                response_id = final_response.id

            if buffer.strip():
                criterion = self._parse_criterion(buffer.strip())
                if criterion:
                    criteria.append(criterion)

            logger.info(f"→ Criteria complete: {len(criteria)} criteria")
            yield {"stage": "criteria", "status": "complete", "data": criteria}

            logger.info("→ Starting general_comment stage")
            yield {"stage": "general_comment", "status": "starting", "data": None}

            stream_manager = await self._llm.analyze_general_comment_streaming(
                previous_response_id=response_id,
                system_prompt=get_general_comment_streaming_prompt(),
            )

            general_comment = ""
            async with stream_manager as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        if hasattr(event, "delta") and event.delta:
                            general_comment += event.delta
                            # logger.info(
                            #     f"→ Streaming general comment delta (buffer: {len(general_comment)} chars)"
                            # )
                            yield {
                                "stage": "general_comment",
                                "status": "streaming",
                                "data": general_comment,
                            }
                    elif event.type == "response.completed":
                        logger.info("→ General comment stream completed")

                final_response = await stream.get_final_response()
                response_id = final_response.id

            logger.info(f"→ General comment complete")
            yield {"stage": "general_comment", "status": "complete", "data": general_comment}

            final_result = AnalysisResult(
                structured_data=StructuredData(
                    complaints=complaints,
                    diagnosis=diagnosis,
                    medications=medications,
                    image_findings=[],
                ),
                prescription_review=PrescriptionReview(
                    status="ok" if not recommendations else "warning",
                    recommendations=recommendations,
                ),
                doctor_evaluation=DoctorEvaluation(
                    criteria=criteria, general_comment=general_comment
                ),
                formatted_transcript=formatted_transcript,
            )

            logger.info("✓ All stages complete! Sending final result")
            yield {"stage": "complete", "status": "complete", "data": final_result}

        except Exception as e:
            logger.error(f"✗ Error during streaming analysis: {e}")
            yield {"stage": "error", "status": "error", "data": str(e)}


def get_streaming_session_service() -> MedicalSessionStreamingService:
    return MedicalSessionStreamingService()
