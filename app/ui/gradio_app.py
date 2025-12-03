import gradio as gr
import pandas as pd

from app.services.session import get_session_service

service = get_session_service()


async def analyze_visit(audio_path: str):
    if not audio_path:
        return "No audio provided", "", pd.DataFrame(), "", None

    transcript, analysis = await service.process_audio(audio_path)

    # Format Recommendations
    recs_text = "\n".join([f"- {r}" for r in analysis.prescription_review.recommendations])
    if not recs_text:
        recs_text = "Рекомендаций нет."

    # Format Evaluation Table
    eval_data = []
    for key, criterion in analysis.doctor_evaluation.criteria.items():
        eval_data.append(
            {"Критерий": key, "Оценка": f"{criterion.score}/5", "Комментарий": criterion.comment}
        )
    eval_df = pd.DataFrame(eval_data)

    # General Comment
    gen_comment = analysis.doctor_evaluation.general_comment

    return transcript, recs_text, eval_df, gen_comment, None  # Clear previous audio


async def voice_recommendations(recs_text: str):
    if not recs_text:
        return None

    # Clean up bullets for TTS
    clean_text = recs_text.replace("- ", "").replace("\n", ". ")
    audio_path = await service.generate_voice_recommendations([clean_text])
    return audio_path


def create_app():
    with gr.Blocks(title="Medical AI Assistant") as app:
        gr.Markdown("## 🏥 Medical AI Assistant Demo")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Запись приема / Загрузка аудио",
                )
                analyze_btn = gr.Button("Начать прием (Анализ)", variant="primary")

            with gr.Column():
                transcript_output = gr.Textbox(label="Транскрипция", lines=10, interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💊 Рекомендации по назначению")
                recs_output = gr.Textbox(label="Список рекомендаций", lines=5, interactive=False)
                voice_btn = gr.Button("🔊 Озвучить рекомендации")
                audio_output = gr.Audio(label="Озвучка", interactive=False, autoplay=True)

            with gr.Column():
                gr.Markdown("### 📋 Оценка работы врача")
                eval_table = gr.Dataframe(
                    label="Чек-лист", headers=["Критерий", "Оценка", "Комментарий"]
                )
                general_comment = gr.Textbox(label="Общий комментарий", lines=3, interactive=False)

        # Actions
        analyze_btn.click(
            fn=analyze_visit,
            inputs=[audio_input],
            outputs=[transcript_output, recs_output, eval_table, general_comment, audio_output],
        )

        voice_btn.click(fn=voice_recommendations, inputs=[recs_output], outputs=[audio_output])

    return app
