import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables (reusing .env handling logic minimally for this script)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("Error: OPENAI_API_KEY not found in .env")
    exit(1)

# Hardcoded script
DIALOGUE_SCRIPT = [
    ("Врач", "alloy", "Добрый день. Проходите, садитесь. На что жалуетесь?"),
    (
        "Пациент",
        "fable",
        "Здравствуйте, доктор. У меня третий день держится температура, около 38. И кашель замучил, сухой такой.",
    ),
    ("Врач", "alloy", "Понятно. Боль в горле, насморк есть?"),
    (
        "Пациент",
        "fable",
        "Горло немного першит, насморка почти нет. Слабость сильная, ломит всё тело.",
    ),
    ("Врач", "alloy", "Скажите, хронические заболевания есть? Аллергия на лекарства?"),
    (
        "Пациент",
        "fable",
        "У меня гастрит был когда-то. Аллергии вроде нет, ну, по крайней мере, я не замечал.",
    ),
    (
        "Врач",
        "alloy",
        "Хорошо. Послушаю вас... Так, дыхание жесткое, хрипов нет. Горло слегка красное. Похоже на ОРВИ. Я выпишу вам противовирусное, Ингавирин, по 90 мг раз в день, 7 дней. И от кашля — Амбробене.",
    ),
    ("Пациент", "fable", "А антибиотики не нужны?"),
    (
        "Врач",
        "alloy",
        "Пока нет, это вирусная инфекция. Если через 3 дня температура не спадет — тогда приходите, будем решать. Пейте больше воды, чай с лимоном.",
    ),
    ("Пациент", "fable", "Хорошо, спасибо большое, доктор."),
    ("Врач", "alloy", "Выздоравливайте. До свидания."),
]

OUTPUT_DIR = Path("sample_audio")
OUTPUT_FILE = OUTPUT_DIR / "dialogue_sample.mp3"


async def generate_dialogue_audio():
    client = AsyncOpenAI(api_key=API_KEY)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Generating dialogue audio to {OUTPUT_FILE}...")

    temp_files = []

    try:
        # Generate audio for each turn
        for idx, (role, voice, text) in enumerate(DIALOGUE_SCRIPT):
            print(f"[{idx+1}/{len(DIALOGUE_SCRIPT)}] {role}: {text[:30]}...")

            temp_file = OUTPUT_DIR / f"part_{idx}_{role}.mp3"
            temp_files.append(temp_file)

            async with client.audio.speech.with_streaming_response.create(
                model="tts-1", voice=voice, input=text
            ) as response:
                await response.stream_to_file(temp_file)

        # Combine audio files using ffmpeg (requires ffmpeg installed)
        # Alternatively, we can just concatenate binary content if MP3 format allows it simply,
        # but proper concatenation usually requires re-encoding or a container tool.
        # For simplicity in python without extra heavy libs like pydub+ffmpeg, we can try simple binary concat for MP3
        # (often works for simple players, though not strictly standard compliant without fixing headers).
        # BUT: let's use pydub if available or just simple binary append for a quick hack script.
        # Given we don't want to force install ffmpeg if user doesn't have it, let's try binary append first.

        print("Combining audio segments...")
        with open(OUTPUT_FILE, "wb") as outfile:
            for fpath in temp_files:
                with open(fpath, "rb") as infile:
                    outfile.write(infile.read())

        print(f"Done! Saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error generating audio: {e}")
    finally:
        # Cleanup temp files
        print("Cleaning up temporary files...")
        for fpath in temp_files:
            if fpath.exists():
                fpath.unlink()


if __name__ == "__main__":
    asyncio.run(generate_dialogue_audio())
