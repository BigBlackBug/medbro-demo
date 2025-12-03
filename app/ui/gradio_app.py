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

    # Structured Data
    complaints = "\n".join([f"- {c}" for c in analysis.structured_data.complaints])
    diagnosis = analysis.structured_data.diagnosis or "Не установлен"
    medications = []
    for m in analysis.structured_data.medications:
        med_str = f"- {m.name}"
        if m.dosage:
            med_str += f" ({m.dosage})"
        if m.frequency:
            med_str += f", {m.frequency}"
        medications.append(med_str)
    meds_text = "\n".join(medications) if medications else "Назначений нет"

    return transcript, recs_text, eval_df, gen_comment, complaints, diagnosis, meds_text


def create_app():
    with gr.Blocks(title="Medical AI Assistant") as app:
        gr.HTML("<style>footer {visibility: hidden}</style>")
        gr.Markdown("## 🏥 Medical AI Assistant Demo")

        # Верхний блок: Аудио и Транскрипция
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Запись приема / Загрузка аудио",
                )
                analyze_btn = gr.Button("Начать прием (Анализ)", variant="primary")

            with gr.Column(scale=1):
                transcript_output = gr.Textbox(label="Транскрипция", lines=10, interactive=False)

        # Средний блок: Структурированные данные
        gr.Markdown("### 📝 Данные приема")
        with gr.Row():
            with gr.Column():
                complaints_output = gr.Textbox(label="Жалобы", lines=5, interactive=False)
            with gr.Column():
                diagnosis_output = gr.Textbox(label="Диагноз", lines=2, interactive=False)
            with gr.Column():
                meds_output = gr.Textbox(label="Назначения", lines=5, interactive=False)

        # Нижний блок: Рекомендации и Оценка
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💊 Клинические рекомендации")
                recs_output = gr.Textbox(
                    label="Рекомендации для врача", lines=10, interactive=False
                )

            with gr.Column():
                gr.Markdown("### 📋 Оценка коммуникации")
                eval_table = gr.Dataframe(
                    label="Чек-лист врача", headers=["Критерий", "Оценка", "Комментарий"]
                )

        # Финальный комментарий
        gr.Markdown("---")
        general_comment = gr.Textbox(label="Общее заключение", lines=3, interactive=False)

        # Actions
        analyze_btn.click(
            fn=analyze_visit,
            inputs=[audio_input],
            outputs=[
                transcript_output,
                recs_output,
                eval_table,
                general_comment,
                complaints_output,
                diagnosis_output,
                meds_output,
            ],
        )

    return app
