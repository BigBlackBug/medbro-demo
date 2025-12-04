import gradio as gr
import pandas as pd

from app.core.models import AnalysisResult, DialogueTurn
from app.services.session import get_session_service
from config.prompts import get_analysis_prompt, SYSTEM_PROMPT_GENERATE_DIALOGUE
from config.logger import logger

service = get_session_service()


def format_transcript_html(content: str) -> str:
    return f"""
    <div style="
        height: 300px; 
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


def format_transcript_plain(transcript: list[DialogueTurn]) -> str:
    content = "<br>".join([f'<b style="color: #000000;">{t.speaker}:</b> {t.text}' for t in transcript])
    return format_transcript_html(content)


def format_transcript_highlighted(analysis: AnalysisResult, transcript: list[DialogueTurn]) -> str:
    if analysis.formatted_transcript:
        content = analysis.formatted_transcript
        if "<br>" not in content:
            content = content.replace("\n", "<br>")
        return format_transcript_html(content)
    return format_transcript_plain(transcript)


async def analyze_visit(audio_path: str):
    if not audio_path:
        yield "No audio provided", "", pd.DataFrame(), "", "", "", ""
        return

    logger.info(f"Starting audio processing for: {audio_path}")
    
    transcript_raw = await service.stt.transcribe(audio_path)
    logger.info(f"Transcription completed: {len(transcript_raw)} turns")
    
    yield (
        format_transcript_plain(transcript_raw),
        "⏳ Analyzing...",
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip()
    )
    
    system_prompt = get_analysis_prompt()
    analysis = await service.llm.analyze(transcript_raw, system_prompt)
    logger.info("Analysis completed")
    
    recs_text = "\n".join([f"- {r}" for r in analysis.prescription_review.recommendations])
    if not recs_text:
        recs_text = "No recommendations."

    eval_data = []
    for criterion in analysis.doctor_evaluation.criteria:
        eval_data.append(
            {
                "Criterion": criterion.name,
                "Score": f"{criterion.score}/5",
                "Comment": criterion.comment,
            }
        )
    eval_df = pd.DataFrame(eval_data)

    gen_comment = analysis.doctor_evaluation.general_comment

    complaints = "\n".join([f"- {c}" for c in analysis.structured_data.complaints])
    diagnosis = analysis.structured_data.diagnosis or "Not established"
    medications = []
    for m in analysis.structured_data.medications:
        med_str = f"- {m.name}"
        if m.dosage:
            med_str += f" ({m.dosage})"
        if m.frequency:
            med_str += f", {m.frequency}"
        medications.append(med_str)
    meds_text = "\n".join(medications) if medications else "No prescriptions"
    
    yield (
        format_transcript_highlighted(analysis, transcript_raw),
        recs_text,
        eval_df,
        gen_comment,
        complaints,
        diagnosis,
        meds_text
    )


async def generate_and_analyze():
    logger.info("Generating sample dialogue...")
    
    generated_dialogue = await service.llm.generate_dialogue(
        SYSTEM_PROMPT_GENERATE_DIALOGUE
    )
    
    transcript_turns = [
        DialogueTurn(speaker=turn.role, text=turn.text) for turn in generated_dialogue.dialogue
    ]
    logger.info(f"Dialogue generation completed: {len(transcript_turns)} turns")
    
    yield (
        format_transcript_plain(transcript_turns),
        "⏳ Analyzing...",
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip()
    )
    
    system_prompt = get_analysis_prompt()
    analysis = await service.llm.analyze(transcript_turns, system_prompt)
    logger.info("Analysis completed")
    
    recs_text = "\n".join([f"- {r}" for r in analysis.prescription_review.recommendations])
    if not recs_text:
        recs_text = "No recommendations."

    eval_data = []
    for criterion in analysis.doctor_evaluation.criteria:
        eval_data.append(
            {
                "Criterion": criterion.name,
                "Score": f"{criterion.score}/5",
                "Comment": criterion.comment,
            }
        )
    eval_df = pd.DataFrame(eval_data)

    gen_comment = analysis.doctor_evaluation.general_comment

    complaints = "\n".join([f"- {c}" for c in analysis.structured_data.complaints])
    diagnosis = analysis.structured_data.diagnosis or "Not established"
    medications = []
    for m in analysis.structured_data.medications:
        med_str = f"- {m.name}"
        if m.dosage:
            med_str += f" ({m.dosage})"
        if m.frequency:
            med_str += f", {m.frequency}"
        medications.append(med_str)
    meds_text = "\n".join(medications) if medications else "No prescriptions"
    
    yield (
        format_transcript_highlighted(analysis, transcript_turns),
        recs_text,
        eval_df,
        gen_comment,
        complaints,
        diagnosis,
        meds_text
    )


async def play_recommendations(recommendations_text: str):
    if not recommendations_text or recommendations_text == "No recommendations.":
        return None
    
    audio_path = await service.text_to_speech(text=recommendations_text, voice="alloy")
    return audio_path


def create_app():
    with gr.Blocks(title="Medical AI Assistant") as app:
        gr.HTML("<style>footer {visibility: hidden}</style>")
        gr.Markdown("## 🏥 Medical AI Assistant Demo")

        # Top Block: Audio and Transcription
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Consultation Recording / Upload Audio",
                )
                with gr.Row():
                    analyze_btn = gr.Button("Start Consultation (Analyze)", variant="primary")
                    generate_btn = gr.Button("Generate Example and Analyze", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 🗣️ Transcription")
                transcript_output = gr.HTML(
                    value='<div style="color: #6b7280; font-style: italic;">Transcription text with key highlights will appear here...</div>'
                )

        # Middle Block: Structured Data
        gr.Markdown("### 📝 Consultation Data")
        with gr.Row():
            with gr.Column():
                complaints_output = gr.Textbox(label="Complaints", lines=5, interactive=False)
            with gr.Column():
                diagnosis_output = gr.Textbox(label="Diagnosis", lines=2, interactive=False)
            with gr.Column():
                meds_output = gr.Textbox(label="Prescriptions", lines=5, interactive=False)

        # Bottom Block: Recommendations and Evaluation
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💊 Clinical Recommendations")
                recs_output = gr.Textbox(
                    label="Recommendations for Doctor", lines=10, interactive=False
                )
                play_recs_btn = gr.Button("🔊 Play Recommendations Audio", variant="secondary", size="sm")
                recs_audio_output = gr.Audio(
                    label="Recommendations Audio", 
                    visible=True, 
                    interactive=False,
                    autoplay=True
                )

            with gr.Column():
                gr.Markdown("### 📋 Communication Evaluation")
                eval_table = gr.Dataframe(
                    label="Doctor Checklist", headers=["Criterion", "Score", "Comment"]
                )

        # Final Comment
        gr.Markdown("---")
        general_comment = gr.Textbox(label="General Conclusion", lines=3, interactive=False)

        # Actions
        outputs_list = [
            transcript_output,
            recs_output,
            eval_table,
            general_comment,
            complaints_output,
            diagnosis_output,
            meds_output,
        ]

        analyze_btn.click(
            fn=analyze_visit,
            inputs=[audio_input],
            outputs=outputs_list,
        )

        generate_btn.click(
            fn=generate_and_analyze,
            inputs=[],
            outputs=outputs_list,
        )

        play_recs_btn.click(
            fn=play_recommendations,
            inputs=[recs_output],
            outputs=[recs_audio_output],
        )

    return app
