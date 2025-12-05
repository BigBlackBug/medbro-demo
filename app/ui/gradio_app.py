from collections.abc import AsyncIterator

import gradio as gr

from app.core.models import AnalysisResult, DialogueTurn, ImageAttachment
from app.services.session import get_session_service
from config.logger import logger
from config.settings import config

service = get_session_service()


def format_transcript_html(content: str) -> str:
    return f"""
    <div style="
        height: 500px;
        overflow-y: auto; 
        padding: 12px; 
        border: 1px solid #e5e7eb; 
        border-radius: 8px; 
        background-color: #ffffff;
        font-family: system-ui, -apple-system, sans-serif;
        line-height: 1.6;
        color: #1f2937;
    ">
        {content}
    </div>
    """


def format_criteria_cards(criteria_list: list) -> str:
    if not criteria_list:
        return "<div>No evaluation data available</div>"

    cards_html = """
    <div style="display: flex; flex-direction: column; gap: 12px; font-family: system-ui, -apple-system, sans-serif;">
    """

    for criterion in criteria_list:
        score = criterion["score"]

        if score >= 4:
            color = "#166534"
            bg_color = "#dcfce7"
        elif score == 3:
            color = "#92400e"
            bg_color = "#fef3c7"
        else:
            color = "#991b1b"
            bg_color = "#fee2e2"

        cards_html += f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            background-color: #ffffff;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 16px; color: #1f2937;">{criterion["name"]}</span>
                <span style="
                    background-color: {bg_color};
                    color: {color};
                    padding: 4px 12px;
                    border-radius: 6px;
                    font-weight: 600;
                    font-size: 14px;
                    font-weight: 600;
                    font-size: 14px;
                ">{score}/5</span>
            </div>
            <div style="color: #4b5563; line-height: 1.5;">{criterion["comment"]}</div>
        </div>
        """

    cards_html += "</div>"
    return cards_html


def format_recommendations_html(recommendations: list[str]) -> str:
    if not recommendations:
        return """
        <div style="
            padding: 16px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background-color: #f9fafb;
            color: #6b7280;
            font-family: system-ui, -apple-system, sans-serif;
            text-align: center;
        ">
            No recommendations.
        </div>
        """

    html = """
    <div style="
        display: flex;
        flex-direction: column;
        gap: 12px;
        font-family: system-ui, -apple-system, sans-serif;
    ">
    """

    for idx, rec in enumerate(recommendations, 1):
        html += f"""
        <div style="
            border-left: 4px solid #4b5563;
            padding: 12px 16px;
            background-color: #f9fafb;
            border-radius: 6px;
        ">
            <div style="
                display: flex;
                gap: 12px;
                align-items: start;
            ">
                <span style="
                    background-color: #4b5563;
                    color: white;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 600;
                    font-size: 14px;
                    flex-shrink: 0;
                ">{idx}</span>
                <div style="color: #1f2937; line-height: 1.6; padding-top: 2px;">{rec}</div>
            </div>
        </div>
        """

    html += "</div>"
    return html


def format_data_card(title: str, content: str, emoji: str = "") -> str:
    display_title = f"{emoji} {title}" if emoji else title

    if not content or content in [
        "Not established",
        "No prescriptions",
        "No images analyzed",
        "No significant findings",
    ]:
        content_color = "#9ca3af"
        content_style = "font-style: italic;"
    else:
        content_color = "#1f2937"
        content_style = ""

    return f"""
    <div style="
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        background-color: #ffffff;
        font-family: system-ui, -apple-system, sans-serif;
        height: 100%;
    ">
        <div style="
            font-weight: 600;
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        ">{display_title}</div>
        <div style="
            color: {content_color};
            line-height: 1.6;
            white-space: pre-wrap;
            {content_style}
        ">{content}</div>
    </div>
    """


def format_general_comment(comment: str) -> str:
    if not comment:
        comment = "No general comment provided."
        color = "#9ca3af"
        style = "font-style: italic;"
    else:
        color = "#1f2937"
        style = ""

    return f"""
    <div style="
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        background-color: #f9fafb;
        font-family: system-ui, -apple-system, sans-serif;
    ">
        <div style="
            font-weight: 600;
            font-size: 16px;
            color: #374151;
            margin-bottom: 12px;
        ">📋 General Conclusion</div>
        <div style="
            color: {color};
            line-height: 1.6;
            {style}
        ">{comment}</div>
    </div>
    """


def format_status(message: str, is_complete: bool = False) -> str:
    if not message:
        return ""

    if is_complete:
        bg_color = "#dcfce7"
        border_color = "#86efac"
        text_color = "#166534"
        icon = "✓"
    else:
        bg_color = "#dbeafe"
        border_color = "#93c5fd"
        text_color = "#1e40af"
        icon = "⏳"

    return f"""
    <div style="
        padding: 12px 16px;
        border-radius: 8px;
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        color: {text_color};
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 14px;
        margin-top: 12px;
    ">
        <span style="font-weight: 600;">{icon}</span> {message}
    </div>
    """


def format_transcript_plain(transcript: list[DialogueTurn]) -> str:
    content = "<br>".join(
        [f'<b style="color: #000000;">{t.speaker}:</b> {t.text}' for t in transcript]
    )
    return format_transcript_html(content)


def format_transcript_highlighted(analysis: AnalysisResult, transcript: list[DialogueTurn]) -> str:
    if analysis.formatted_transcript:
        content = analysis.formatted_transcript
        if "<br>" not in content:
            content = content.replace("\n", "<br>")
        return format_transcript_html(content)
    return format_transcript_plain(transcript)


async def analyze_visit(audio_path: str, images: list | None) -> AsyncIterator[tuple]:
    loading_html = (
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>⏳ Loading...</div>"
    )
    if not audio_path:
        yield (
            "No audio provided",
            "<div>No audio provided</div>",
            "",
            "<div>No data</div>",
            loading_html,
            loading_html,
            loading_html,
            loading_html,
            loading_html,
            format_status("No audio provided", False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    logger.info(f"Starting audio processing for: {audio_path}")
    yield (
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>Processing...</div>",
        loading_html,
        "",
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        format_status("Starting parallel processing (STT + Images)...", False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

    # Prepare image attachments
    image_attachments: list[ImageAttachment] | None = None
    if images:
        image_attachments = []
        for img_path in images:
            if isinstance(img_path, str):
                image_attachments.append(ImageAttachment(file_path=img_path))
            elif hasattr(img_path, "name"):
                image_attachments.append(ImageAttachment(file_path=img_path.name))
        logger.info(f"Processing {len(image_attachments)} image(s)")

    # Parallel Execution via Service
    transcript_raw, image_report = await service.process_upload(audio_path, image_attachments)

    logger.info(f"Transcription completed: {len(transcript_raw)} turns")
    if image_report:
        logger.info("Image analysis completed")

    yield (
        format_transcript_plain(transcript_raw),
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>⏳ Analyzing...</div>",
        "",
        "<div style='padding: 20px; text-align: center;'>⏳ Loading...</div>",
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        format_status("Transcription complete. Analyzing consultation...", False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

    analysis = await service.analyze_consultation(transcript_raw, image_report)
    logger.info("Analysis completed")

    recs_html = format_recommendations_html(analysis.prescription_review.recommendations)
    recs_text = (
        "\n\n".join(analysis.prescription_review.recommendations)
        if analysis.prescription_review.recommendations
        else ""
    )

    criteria_data = []
    for criterion in analysis.doctor_evaluation.criteria:
        criteria_data.append(
            {
                "name": criterion.name,
                "score": criterion.score,
                "comment": criterion.comment,
            }
        )
    eval_html = format_criteria_cards(criteria_data)

    gen_comment_html = format_general_comment(analysis.doctor_evaluation.general_comment)

    complaints_text = "\n".join([f"- {c}" for c in analysis.structured_data.complaints])
    if not complaints_text:
        complaints_text = "No complaints recorded"
    complaints_html = format_data_card(title="Complaints", content=complaints_text)

    diagnosis_text = analysis.structured_data.diagnosis or "Not established"
    diagnosis_html = format_data_card(title="Diagnosis", content=diagnosis_text)

    medications = []
    for m in analysis.structured_data.medications:
        med_str = f"- {m.name}"
        if m.dosage:
            med_str += f" ({m.dosage})"
        if m.frequency:
            med_str += f", {m.frequency}"
        medications.append(med_str)
    meds_text = "\n".join(medications) if medications else "No prescriptions"
    meds_html = format_data_card(title="Medications", content=meds_text)

    image_findings_text = "\n".join([f"- {f}" for f in analysis.structured_data.image_findings])
    if not image_findings_text:
        image_findings_text = "No images analyzed" if not images else "No significant findings"
    image_findings_html = format_data_card(
        title="Image Analysis Findings", content=image_findings_text, emoji="🔬"
    )

    has_recs = bool(recs_text and recs_text.strip())
    yield (
        format_transcript_highlighted(analysis, transcript_raw),
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
    )


async def generate_and_analyze(
    diagnosis: str | None, doctor_skill: int, images: list | None
) -> AsyncIterator[tuple]:
    diagnosis = diagnosis.strip() if diagnosis else None
    if not diagnosis:
        diagnosis = None

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
        loading_html,
        format_status("Generating dialogue and analyzing images...", False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

    # Prepare image attachments
    image_attachments: list[ImageAttachment] | None = None
    if images:
        image_attachments = []
        for img_path in images:
            if isinstance(img_path, str):
                image_attachments.append(ImageAttachment(file_path=img_path))
            elif hasattr(img_path, "name"):
                image_attachments.append(ImageAttachment(file_path=img_path.name))
        logger.info(f"Processing {len(image_attachments)} image(s)")

    # Parallel Execution via Service
    transcript_turns, image_report = await service.generate_simulation(
        diagnosis, doctor_skill, image_attachments
    )

    logger.info(f"Dialogue generation completed: {len(transcript_turns)} turns")

    yield (
        format_transcript_plain(transcript_turns),
        "<div style='padding: 20px; text-align: center; color: #6b7280;'>⏳ Analyzing...</div>",
        "",
        "<div style='padding: 20px; text-align: center;'>⏳ Loading...</div>",
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        loading_html,
        format_status("Dialogue generated. Analyzing consultation...", False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

    analysis = await service.analyze_consultation(transcript_turns, image_report)
    logger.info("Analysis completed")
    recs_html = format_recommendations_html(analysis.prescription_review.recommendations)
    recs_text = (
        "\n\n".join(analysis.prescription_review.recommendations)
        if analysis.prescription_review.recommendations
        else ""
    )

    criteria_data = []
    for criterion in analysis.doctor_evaluation.criteria:
        criteria_data.append(
            {
                "name": criterion.name,
                "score": criterion.score,
                "comment": criterion.comment,
            }
        )
    eval_html = format_criteria_cards(criteria_data)

    gen_comment_html = format_general_comment(analysis.doctor_evaluation.general_comment)

    complaints_text = "\n".join([f"- {c}" for c in analysis.structured_data.complaints])
    if not complaints_text:
        complaints_text = "No complaints recorded"
    complaints_html = format_data_card(title="Complaints", content=complaints_text)

    diagnosis_text = analysis.structured_data.diagnosis or "Not established"
    diagnosis_html = format_data_card(title="Diagnosis", content=diagnosis_text)

    medications = []
    for m in analysis.structured_data.medications:
        med_str = f"- {m.name}"
        if m.dosage:
            med_str += f" ({m.dosage})"
        if m.frequency:
            med_str += f", {m.frequency}"
        medications.append(med_str)
    meds_text = "\n".join(medications) if medications else "No prescriptions"
    meds_html = format_data_card(title="Medications", content=meds_text)

    image_findings_text = "\n".join([f"- {f}" for f in analysis.structured_data.image_findings])
    if not image_findings_text:
        image_findings_text = "No images analyzed" if not images else "No significant findings"
    image_findings_html = format_data_card(
        title="Image Analysis Findings", content=image_findings_text, emoji="🔬"
    )

    has_recs = bool(recs_text and recs_text.strip())
    yield (
        format_transcript_highlighted(analysis, transcript_turns),
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
    )


async def play_recommendations(recommendations_text: str) -> tuple[str | None, dict]:
    if not recommendations_text or not recommendations_text.strip():
        logger.warning("No recommendations to play")
        return None, gr.update(interactive=True, value="🔊 Play Recommendations Audio")

    try:
        logger.info("Generating recommendations audio...")
        audio_path = await service.text_to_speech(
            text=recommendations_text, voice=config.DEFAULT_TTS_VOICE
        )
        logger.info(f"Recommendations audio ready: {audio_path}")
        return audio_path, gr.update(interactive=True, value="🔊 Play Recommendations Audio")
    except Exception as e:
        logger.error(f"Error generating recommendations audio: {e}")
        return None, gr.update(interactive=True, value="🔊 Play Recommendations Audio")


def toggle_analyze_button(audio_path: str | None) -> dict:
    if audio_path and audio_path.strip():
        return gr.update(interactive=True)
    return gr.update(interactive=False)


async def generate_dialogue_audio(
    diagnosis: str | None, doctor_skill: int
) -> AsyncIterator[tuple[str | None, dict, str]]:
    try:
        yield None, gr.update(interactive=False), format_status(
            "Generating dialogue audio...", False
        )

        output_file = await service.generate_dialogue_audio(
            diagnosis=diagnosis, doctor_skill=doctor_skill, output_dir=config.DATA_DIR
        )

        yield output_file, gr.update(interactive=True), format_status(
            "Audio generation complete!", True
        )

    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        yield None, gr.update(interactive=True), format_status(f"Error: {str(e)}", False)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Medical AI Assistant") as app:
        gr.HTML("<style>footer {visibility: hidden}</style>")
        gr.Markdown("## 🏥 Medical AI Assistant Demo")

        status_output = gr.HTML(value="", visible=True)

        # Top Block: Input Tabs and Transcription
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

                    with gr.Tab("🎭 Generate Example"):
                        diagnosis_input = gr.Textbox(
                            label="Diagnosis (optional)",
                            placeholder="e.g., bronchitis, flu, pneumonia... Leave empty for random",
                            lines=1,
                        )
                        doctor_skill_input = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Doctor's Skill Level",
                            info="0=Novice, 5=Competent (2 years exp), 10=Expert Master",
                        )
                        images_input_generate = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="📷 Medical Images (X-rays, Lab Reports, Prescriptions)",
                            type="filepath",
                        )
                        generate_btn = gr.Button(
                            "Generate Example and Analyze", variant="secondary", size="lg"
                        )

                    with gr.Tab("🎵 Generate Audio"):
                        diagnosis_audio_input = gr.Textbox(
                            label="Diagnosis (optional)",
                            placeholder="e.g., bronchitis, flu, pneumonia... Leave empty for random",
                            lines=1,
                        )
                        doctor_skill_audio_input = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Doctor's Skill Level",
                            info="0=Novice, 5=Competent (2 years exp), 10=Expert Master",
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

        # Separator
        gr.Markdown("---")

        # Middle Block: Structured Data
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

        with gr.Row():
            with gr.Column():
                image_findings_output = gr.HTML(
                    value="<div style='color: #6b7280; font-style: italic;'>Image findings will appear here...</div>"
                )

        # Separator
        gr.Markdown("---")

        # Bottom Block: Recommendations and Evaluation
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

        # Separator
        gr.Markdown("---")

        # Final Comment
        general_comment = gr.HTML(
            value="<div style='color: #6b7280; font-style: italic;'>General conclusion will appear here...</div>"
        )

        # Actions
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

        audio_input.change(fn=toggle_analyze_button, inputs=[audio_input], outputs=[analyze_btn])

        analyze_btn.click(
            fn=analyze_visit,
            inputs=[audio_input, images_input],
            outputs=outputs_list
            + [status_output, audio_input, images_input, analyze_btn, play_recs_btn],
        )

        generate_btn.click(
            fn=generate_and_analyze,
            inputs=[diagnosis_input, doctor_skill_input, images_input_generate],
            outputs=outputs_list
            + [
                status_output,
                diagnosis_input,
                doctor_skill_input,
                images_input_generate,
                generate_btn,
                play_recs_btn,
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
