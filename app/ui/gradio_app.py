import gradio as gr
import pandas as pd

from app.core.models import AnalysisResult, DialogueTurn
from app.services.session import get_session_service

service = get_session_service()


def format_results(transcript: str | list[DialogueTurn], analysis: AnalysisResult):
    # Format Recommendations
    recs_text = "\n".join([f"- {r}" for r in analysis.prescription_review.recommendations])
    if not recs_text:
        recs_text = "No recommendations."

    # Format Evaluation Table
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

    # General Comment
    gen_comment = analysis.doctor_evaluation.general_comment

    # Structured Data
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

    # Format Transcript with Highlights
    if analysis.formatted_transcript:
        transcript_content = analysis.formatted_transcript
    else:
        if isinstance(transcript, list):
            transcript_content = "<br>".join([f"<b>{t.speaker}:</b> {t.text}" for t in transcript])
        else:
            transcript_content = str(transcript)

    # Ensure newlines are preserved in HTML if not already <br>
    if "<br>" not in transcript_content:
        transcript_content = transcript_content.replace("\n", "<br>")

    formatted_transcript = f"""
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
        {transcript_content}
    </div>
    """

    return formatted_transcript, recs_text, eval_df, gen_comment, complaints, diagnosis, meds_text


async def analyze_visit(audio_path: str):
    if not audio_path:
        return "No audio provided", "", pd.DataFrame(), "", "", "", ""

    transcript, analysis = await service.process_audio(audio_path)
    return format_results(transcript, analysis)


async def generate_and_analyze():
    transcript, analysis = await service.generate_and_analyze_sample()
    return format_results(transcript, analysis)


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
