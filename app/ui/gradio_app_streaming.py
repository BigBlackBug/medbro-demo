import asyncio
from collections.abc import AsyncIterator

import gradio as gr

from app.core.models import DialogueTurn, GeneratedDialogue, ImageAttachment
from app.services.session_streaming import get_streaming_session_service
from app.ui.gradio_app import (
    format_criteria_cards,
    format_data_card,
    format_general_comment,
    format_recommendations_html,
    format_status,
    format_transcript_html,
    generate_dialogue_audio,
    play_recommendations,
)
from config.logger import logger
from config.prompts import get_dialogue_generation_prompt


def format_markdown_card(content: str) -> str:
    if not content or content in [
        "Not established",
        "No images analyzed",
        "No significant findings",
    ]:
        content = f"*{content}*"

    return f"{content}"


streaming_service = get_streaming_session_service()


def format_transcript_highlighted_streaming(formatted_transcript: str) -> str:
    if formatted_transcript:
        content = formatted_transcript
        if "<br>" not in content:
            content = content.replace("\n", "<br>")
        return format_transcript_html(content)
    return "<div style='padding: 20px; text-align: center; color: #6b7280;'>⏳ Loading...</div>"


async def analyze_images_only(images: list) -> AsyncIterator[tuple]:
    loading_html = (
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>⏳ Loading...</div>"
    )
    empty_html = "<div style='padding: 20px; text-align: center; color: #9ca3af; font-style: italic;'>N/A - No consultation audio provided</div>"

    image_attachments: list[ImageAttachment] = []
    for img_path in images:
        if isinstance(img_path, str):
            image_attachments.append(ImageAttachment(file_path=img_path))
        elif hasattr(img_path, "name"):
            image_attachments.append(ImageAttachment(file_path=img_path.name))

    logger.info(f"Analyzing {len(image_attachments)} image(s) without audio...")

    yield (
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>No audio provided - analyzing images only...</div>",
        empty_html,
        "",
        empty_html,
        empty_html,
        empty_html,
        empty_html,
        empty_html,
        "*⏳ Loading...*",
        format_status(f"📸 Analyzing {len(image_attachments)} image(s)...", False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=False),
    )

    image_report = await streaming_service.analyze_images_only(image_attachments)

    image_findings_html = format_markdown_card(content=image_report)

    yield (
        "<div style='padding: 20px; text-align: center; color: #2563eb;'>✓ Image analysis complete (no audio consultation provided)</div>",
        empty_html,
        "",
        empty_html,
        empty_html,
        empty_html,
        empty_html,
        empty_html,
        image_findings_html,
        format_status("✅ Image analysis complete!", True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(open=True),
    )


async def analyze_visit_streaming(audio_path: str, images: list | None) -> AsyncIterator[tuple]:
    loading_html = (
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>⏳ Loading...</div>"
    )

    if not audio_path:
        if images and len(images) > 0:
            async for result in analyze_images_only(images):
                yield result
            return
        else:
            yield (
                "<div style='padding: 20px; text-align: center; color: #dc2626;'>⚠ No audio or images provided</div>",
                loading_html,
                "",
                loading_html,
                loading_html,
                loading_html,
                loading_html,
                loading_html,
                "*⏳ Loading...*",
                format_status("No audio or images provided", False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(open=False),
            )
            return

    image_attachments: list[ImageAttachment] | None = None
    if images:
        image_attachments = []
        for img_path in images:
            if isinstance(img_path, str):
                image_attachments.append(ImageAttachment(file_path=img_path))
            elif hasattr(img_path, "name"):
                image_attachments.append(ImageAttachment(file_path=img_path.name))
        logger.info(f"Processing {len(image_attachments)} image(s)")

    status_text = "🎤 Transcribing audio"
    if image_attachments:
        status_text += f" & 📸 Analyzing {len(image_attachments)} image(s)"
    status_text += "..."

    logger.info(f"Starting streaming audio processing for: {audio_path}")
    yield (
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>Processing...</div>",
        loading_html,
        "",
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        "*⏳ Loading...*",
        format_status(status_text, False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=False),
    )

    transcript_task = asyncio.create_task(streaming_service._stt.transcribe(audio_path))

    image_task = None
    if image_attachments:
        image_task = asyncio.create_task(streaming_service._llm.analyze_images(image_attachments))

    transcript_raw = await transcript_task
    logger.info(f"Transcription completed: {len(transcript_raw)} turns")

    raw_transcript_text = "\n".join([f"{turn.speaker}: {turn.text}" for turn in transcript_raw])
    transcript_display = f"""
    <div style="
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        background-color: #ffffff;
        font-family: system-ui, -apple-system, sans-serif;
    ">
        <div style="font-weight: 600; color: #1f2937; margin-bottom: 12px;">📝 Raw Transcript</div>
        <pre style="
            white-space: pre-wrap;
            font-family: monospace;
            margin: 0;
            color: #1f2937;
            background-color: #f9fafb;
            padding: 12px;
            border-radius: 4px;
        ">{raw_transcript_text}</pre>
    </div>
    """

    status_after_transcript = "✅ Transcription complete"
    if image_task:
        status_after_transcript += " | ⏳ Still processing images..."
    else:
        status_after_transcript += " | 🔄 Starting analysis..."

    yield (
        transcript_display,
        loading_html,
        "",
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        "*⏳ Loading...*",
        format_status(status_after_transcript, False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=False),
    )

    image_report: str | None = None
    image_findings_html = "*⏳ Loading...*"
    if image_task:
        image_report = await image_task
        logger.info("Image analysis completed")
        image_findings_html = format_markdown_card(content=image_report)

        yield (
            transcript_display,
            loading_html,
            "",
            loading_html,
            loading_html,
            loading_html,
            loading_html,
            loading_html,
            image_findings_html,
            format_status("✅ Images analyzed | 🔄 Starting consultation analysis...", False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(open=False),
        )

    transcript_html = loading_html
    recs_html = loading_html
    recs_text = ""
    eval_html = loading_html
    gen_comment_html = loading_html
    complaints_html = loading_html
    diagnosis_html = loading_html
    meds_html = loading_html

    async for update in streaming_service.analyze_consultation_streaming(
        transcript_raw, image_report
    ):
        stage = update.get("stage")
        status = update.get("status")
        data = update.get("data")

        if stage == "transcript":
            if status == "streaming" and data:
                transcript_html = format_transcript_highlighted_streaming(data)
                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("🔍 Formatting and highlighting transcript...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )
            elif status == "complete" and data:
                transcript_html = format_transcript_highlighted_streaming(data)

        elif stage == "complaints":
            if status == "streaming" and data:
                complaints_text = "\n".join([f"- {c}" for c in data])
                if not complaints_text:
                    complaints_text = "No complaints recorded"
                complaints_html = format_data_card(title="Complaints", content=complaints_text)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("📝 Extracting patient complaints...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "diagnosis":
            if status == "streaming" and data:
                diagnosis_text = data or "Not established"
                diagnosis_html = format_data_card(title="Diagnosis", content=diagnosis_text)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("🩺 Identifying diagnosis...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "medications":
            if status == "streaming" and data:
                medications = []
                for m in data:
                    med_str = f"- {m.name}"
                    if m.dosage:
                        med_str += f" ({m.dosage})"
                    if m.frequency:
                        med_str += f", {m.frequency}"
                    medications.append(med_str)
                meds_text = "\n".join(medications) if medications else "No prescriptions"
                meds_html = format_data_card(title="Medications", content=meds_text)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("💊 Extracting prescribed medications...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "recommendations":
            if status == "streaming" and data:
                recs_html = format_recommendations_html(data)
                recs_text = "\n\n".join(data) if data else ""

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status(
                        f"⚠️ Generating clinical recommendations... ({len(data)} so far)", False
                    ),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "criteria":
            if status == "streaming" and data:
                criteria_data = []
                for criterion in data:
                    criteria_data.append(
                        {
                            "name": criterion.name,
                            "score": criterion.score,
                            "comment": criterion.comment,
                        }
                    )
                eval_html = format_criteria_cards(criteria_data)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status(
                        f"📊 Evaluating doctor performance... ({len(data)} criteria)", False
                    ),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "general_comment":
            if status == "streaming" and data:
                gen_comment_html = format_general_comment(data)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("✍️ Writing general evaluation...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "complete":
            has_recs = bool(recs_text and recs_text.strip())
            yield (
                transcript_html,
                recs_html,
                recs_text,
                eval_html,
                gen_comment_html,
                complaints_html,
                diagnosis_html,
                meds_html,
                image_findings_html,
                format_status("Analysis complete!", True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=has_recs),
                gr.update(open=False),
            )

        elif stage == "error":
            yield (
                transcript_html,
                recs_html,
                recs_text,
                eval_html,
                gen_comment_html,
                complaints_html,
                diagnosis_html,
                meds_html,
                image_findings_html,
                format_status(f"Error: {data}", False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(open=False),
            )


async def generate_and_analyze_streaming(
    diagnosis: str | None, doctor_skill: int, images: list | None
) -> AsyncIterator[tuple]:
    diagnosis = diagnosis.strip() if diagnosis else None
    if not diagnosis:
        diagnosis = None

    image_attachments: list[ImageAttachment] | None = None
    if images:
        image_attachments = []
        for img_path in images:
            if isinstance(img_path, str):
                image_attachments.append(ImageAttachment(file_path=img_path))
            elif hasattr(img_path, "name"):
                image_attachments.append(ImageAttachment(file_path=img_path.name))
        logger.info(f"Processing {len(image_attachments)} image(s)")

    status_text = "🎭 Generating dialogue"
    if image_attachments:
        status_text += f" & 📸 Analyzing {len(image_attachments)} image(s)"
    status_text += "..."

    logger.info(f"Generating sample dialogue (diagnosis: {diagnosis}, skill: {doctor_skill})...")

    loading_html = (
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>⏳ Loading...</div>"
    )
    yield (
        loading_html,
        loading_html,
        "",
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        "*⏳ Loading...*",
        format_status(status_text, False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=False),
    )

    system_prompt = get_dialogue_generation_prompt(diagnosis=diagnosis, doctor_skill=doctor_skill)
    dialogue_task = asyncio.create_task(
        streaming_service._llm.generate_dialogue(system_prompt=system_prompt, diagnosis=diagnosis)
    )

    image_task = None
    if image_attachments:
        image_task = asyncio.create_task(streaming_service._llm.analyze_images(image_attachments))

    generated_dialogue: GeneratedDialogue = await dialogue_task
    logger.info(f"Dialogue generation completed: {len(generated_dialogue.dialogue)} turns")

    transcript_turns = [
        DialogueTurn(speaker=turn.role, text=turn.text) for turn in generated_dialogue.dialogue
    ]

    raw_transcript_text = "\n".join([f"{turn.speaker}: {turn.text}" for turn in transcript_turns])
    transcript_display = f"""
    <div style="
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        background-color: #ffffff;
        font-family: system-ui, -apple-system, sans-serif;
    ">
        <div style="font-weight: 600; color: #1f2937; margin-bottom: 12px;">📝 Raw Transcript</div>
        <pre style="
            white-space: pre-wrap;
            font-family: monospace;
            margin: 0;
            color: #1f2937;
            background-color: #f9fafb;
            padding: 12px;
            border-radius: 4px;
        ">{raw_transcript_text}</pre>
    </div>
    """

    status_after_dialogue = "✅ Dialogue generated"
    if image_task:
        status_after_dialogue += " | ⏳ Still processing images..."
    else:
        status_after_dialogue += " | 🔄 Starting analysis..."

    yield (
        transcript_display,
        loading_html,
        "",
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        "*⏳ Loading...*",
        format_status(status_after_dialogue, False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(open=False),
    )

    image_report: str | None = None
    image_findings_html = "*⏳ Loading...*"
    if image_task:
        image_report = await image_task
        logger.info("Image analysis completed")
        image_findings_html = format_markdown_card(content=image_report)

        yield (
            transcript_display,
            loading_html,
            "",
            loading_html,
            loading_html,
            loading_html,
            loading_html,
            loading_html,
            image_findings_html,
            format_status("✅ Images analyzed | 🔄 Starting consultation analysis...", False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(open=False),
        )

    transcript_html = loading_html
    recs_html = loading_html
    recs_text = ""
    eval_html = loading_html
    gen_comment_html = loading_html
    complaints_html = loading_html
    diagnosis_html = loading_html
    meds_html = loading_html

    async for update in streaming_service.analyze_consultation_streaming(
        transcript_turns, image_report
    ):
        stage = update.get("stage")
        status = update.get("status")
        data = update.get("data")

        if stage == "transcript":
            if status == "streaming" and data:
                transcript_html = format_transcript_highlighted_streaming(data)
                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("🔍 Formatting and highlighting transcript...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )
            elif status == "complete" and data:
                transcript_html = format_transcript_highlighted_streaming(data)

        elif stage == "complaints":
            if status == "streaming" and data:
                complaints_text = "\n".join([f"- {c}" for c in data])
                if not complaints_text:
                    complaints_text = "No complaints recorded"
                complaints_html = format_data_card(title="Complaints", content=complaints_text)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("📝 Extracting patient complaints...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "diagnosis":
            if status == "streaming" and data:
                diagnosis_text = data or "Not established"
                diagnosis_html = format_data_card(title="Diagnosis", content=diagnosis_text)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("🩺 Identifying diagnosis...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "medications":
            if status == "streaming" and data:
                medications = []
                for m in data:
                    med_str = f"- {m.name}"
                    if m.dosage:
                        med_str += f" ({m.dosage})"
                    if m.frequency:
                        med_str += f", {m.frequency}"
                    medications.append(med_str)
                meds_text = "\n".join(medications) if medications else "No prescriptions"
                meds_html = format_data_card(title="Medications", content=meds_text)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("💊 Extracting prescribed medications...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "recommendations":
            if status == "streaming" and data:
                recs_html = format_recommendations_html(data)
                recs_text = "\n\n".join(data) if data else ""

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status(
                        f"⚠️ Generating clinical recommendations... ({len(data)} so far)", False
                    ),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "criteria":
            if status == "streaming" and data:
                criteria_data = []
                for criterion in data:
                    criteria_data.append(
                        {
                            "name": criterion.name,
                            "score": criterion.score,
                            "comment": criterion.comment,
                        }
                    )
                eval_html = format_criteria_cards(criteria_data)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status(
                        f"📊 Evaluating doctor performance... ({len(data)} criteria)", False
                    ),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "general_comment":
            if status == "streaming" and data:
                gen_comment_html = format_general_comment(data)

                yield (
                    transcript_html,
                    recs_html,
                    recs_text,
                    eval_html,
                    gen_comment_html,
                    complaints_html,
                    diagnosis_html,
                    meds_html,
                    image_findings_html,
                    format_status("✍️ Writing general evaluation...", False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(open=False),
                )

        elif stage == "complete":
            has_recs = bool(recs_text and recs_text.strip())
            yield (
                transcript_html,
                recs_html,
                recs_text,
                eval_html,
                gen_comment_html,
                complaints_html,
                diagnosis_html,
                meds_html,
                image_findings_html,
                format_status("Analysis complete!", True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=has_recs),
                gr.update(open=False),
            )

        elif stage == "error":
            yield (
                transcript_html,
                recs_html,
                recs_text,
                eval_html,
                gen_comment_html,
                complaints_html,
                diagnosis_html,
                meds_html,
                image_findings_html,
                format_status(f"Error: {data}", False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(open=False),
            )


def toggle_analyze_button(audio_path: str | None, images: list | None) -> tuple[dict, dict]:
    has_audio = audio_path and audio_path.strip()
    has_images = images and len(images) > 0

    if has_audio or has_images:
        if has_audio and has_images:
            button_label = "Analyze Conversation and Images"
        elif has_audio:
            button_label = "Analyze Conversation"
        else:
            button_label = "Analyze Images"
        return gr.update(interactive=True, value=button_label), gr.update()
    return gr.update(interactive=False, value="Analyze Conversation"), gr.update()


def create_streaming_app() -> gr.Blocks:
    with gr.Blocks(title="Medical AI Assistant - Streaming") as app:
        gr.HTML(
            """<style>
footer {visibility: hidden}
.image-findings-markdown {
    background-color: #f9fafb !important;
    padding: 16px !important;
    border-radius: 8px !important;
}
.image-findings-markdown,
.image-findings-markdown .prose,
.image-findings-markdown .markdown-body,
.image-findings-markdown > div,
.image-findings-markdown p,
.image-findings-markdown ul,
.image-findings-markdown ol,
.image-findings-markdown li,
.image-findings-markdown h1,
.image-findings-markdown h2,
.image-findings-markdown h3,
.image-findings-markdown h4,
.image-findings-markdown strong,
.image-findings-markdown em,
.image-findings-markdown * {
    font-size: 16px !important;
    line-height: 1.4 !important;
    color: #1f2937 !important;
}
</style>"""
        )
        gr.Markdown("## 🏥 Medical AI Assistant Demo (Streaming Mode)")

        status_output = gr.HTML(value="", visible=True)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("📁 Upload Recording"):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Consultation Recording / Upload Audio",
                        )
                        images_input = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="📷 Medical Images (X-rays, Lab Reports, Prescriptions)",
                            type="filepath",
                        )
                        analyze_btn = gr.Button(
                            "Start Consultation", variant="primary", size="lg", interactive=False
                        )

                    with gr.Tab("🎭 Generate Dialogue"):
                        diagnosis_input = gr.Textbox(
                            label="Diagnosis (optional)",
                            placeholder="e.g., bronchitis, flu, pneumonia... Leave empty for random",
                            lines=1,
                        )
                        doctor_skill_input = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Doctor's Skill Level",
                            info="1=Novice, 2=Junior, 3=Competent, 4=Proficient, 5=Expert",
                        )
                        images_input_generate = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="📷 Medical Images (X-rays, Lab Reports, Prescriptions)",
                            type="filepath",
                        )
                        generate_btn = gr.Button(
                            "Generate Dialogue and Analyze", variant="secondary", size="lg"
                        )

                    with gr.Tab("🎵 Generate Audio"):
                        diagnosis_audio_input = gr.Textbox(
                            label="Diagnosis (optional)",
                            placeholder="e.g., bronchitis, flu, pneumonia... Leave empty for random",
                            lines=1,
                        )
                        doctor_skill_audio_input = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Doctor's Skill Level",
                            info="1=Novice, 2=Junior, 3=Competent, 4=Proficient, 5=Expert",
                        )
                        generate_audio_btn = gr.Button(
                            "Generate Dialogue Audio", variant="secondary", size="lg"
                        )
                        audio_status = gr.HTML(value="", visible=True)
                        generated_audio_output = gr.Audio(
                            label="Generated Dialogue Audio",
                            visible=True,
                            interactive=False,
                            type="filepath",
                        )

            with gr.Column(scale=1):
                gr.Markdown("### 🗣️ Transcription")
                transcript_output = gr.HTML(
                    value='<div style="color: #6b7280; font-style: italic;">Transcription text with key highlights will appear here...</div>',
                    elem_style={"height": "100%", "display": "flex", "flex-direction": "column"},
                )

        gr.Markdown("---")

        with gr.Accordion("🔬 Image Analysis Findings", open=False) as image_accordion:
            image_findings_output = gr.Markdown(
                value="*Image findings will appear here...*", elem_classes="image-findings-markdown"
            )

        gr.Markdown("---")

        gr.Markdown("### 📝 Consultation Data")
        with gr.Row():
            with gr.Column():
                complaints_output = gr.HTML(
                    value="<div style='color: #6b7280; font-style: italic;'>Complaints will appear here...</div>"
                )
            with gr.Column():
                diagnosis_output = gr.HTML(
                    value="<div style='color: #6b7280; font-style: italic;'>Diagnosis will appear here...</div>"
                )
            with gr.Column():
                meds_output = gr.HTML(
                    value="<div style='color: #6b7280; font-style: italic;'>Medications will appear here...</div>"
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💊 Clinical Recommendations")
                recs_output = gr.HTML(
                    value="<div style='color: #6b7280; font-style: italic;'>Recommendations will appear here...</div>"
                )
                recs_text_state = gr.State(value="")
                play_recs_btn = gr.Button(
                    "🔊 Play Recommendations Audio", variant="secondary", size="sm"
                )
                recs_audio_output = gr.Audio(
                    label="Recommendations Audio", visible=True, interactive=False, autoplay=True
                )

            with gr.Column():
                gr.Markdown("### 📋 Communication Evaluation")
                eval_table = gr.HTML(
                    value="<div style='color: #6b7280; font-style: italic;'>Evaluation results will appear here...</div>"
                )

        gr.Markdown("---")

        general_comment = gr.HTML(
            value="<div style='color: #6b7280; font-style: italic;'>General conclusion will appear here...</div>"
        )

        outputs_list = [
            transcript_output,
            recs_output,
            recs_text_state,
            eval_table,
            general_comment,
            complaints_output,
            diagnosis_output,
            meds_output,
            image_findings_output,
        ]

        outputs_list_with_accordion = outputs_list + [
            status_output,
            audio_input,
            images_input,
            analyze_btn,
            play_recs_btn,
            image_accordion,
        ]

        audio_input.change(
            fn=toggle_analyze_button,
            inputs=[audio_input, images_input],
            outputs=[analyze_btn, audio_input],
        )
        images_input.change(
            fn=toggle_analyze_button,
            inputs=[audio_input, images_input],
            outputs=[analyze_btn, images_input],
        )

        analyze_btn.click(
            fn=analyze_visit_streaming,
            inputs=[audio_input, images_input],
            outputs=outputs_list_with_accordion,
        )

        generate_btn.click(
            fn=generate_and_analyze_streaming,
            inputs=[diagnosis_input, doctor_skill_input, images_input_generate],
            outputs=outputs_list
            + [
                status_output,
                diagnosis_input,
                doctor_skill_input,
                images_input_generate,
                generate_btn,
                play_recs_btn,
                image_accordion,
            ],
        )

        play_recs_btn.click(
            fn=play_recommendations,
            inputs=[recs_text_state],
            outputs=[recs_audio_output, play_recs_btn],
            show_progress="full",
        )

        generate_audio_btn.click(
            fn=generate_dialogue_audio,
            inputs=[diagnosis_audio_input, doctor_skill_audio_input],
            outputs=[generated_audio_output, generate_audio_btn, audio_status],
            show_progress="full",
        )

    return app
