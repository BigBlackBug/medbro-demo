import argparse
import asyncio
import re
import sys
from pathlib import Path

# Add project root to path to allow imports from app
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.services.llm import get_llm_provider
from config.prompts import get_dialogue_generation_prompt
from config.settings import config

# Load environment variables (reusing .env handling logic minimally for this script)
load_dotenv()
API_KEY = config.OPENAI_API_KEY

if not API_KEY:
    print("Error: OPENAI_API_KEY not found in .env")
    exit(1)


OUTPUT_DIR = config.DATA_DIR


def sanitize_diagnosis_for_filename(diagnosis: str) -> str:
    slug = diagnosis.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '_', slug)
    slug = slug.strip('_')
    return slug


def get_next_file_index() -> int:
    existing_files = list(OUTPUT_DIR.glob("*_dialogue_*.mp3"))
    if not existing_files:
        return 1

    max_index = 0
    for file in existing_files:
        try:
            index = int(file.name.split("_")[0])
            max_index = max(max_index, index)
        except (ValueError, IndexError):
            continue

    return max_index + 1


async def generate_dialogue_audio(diagnosis: str | None = None, doctor_skill: int = 5) -> None:
    client = AsyncOpenAI(api_key=API_KEY)
    llm = get_llm_provider()

    OUTPUT_DIR.mkdir(exist_ok=True)

    file_index = get_next_file_index()
    diagnosis_slug: str = sanitize_diagnosis_for_filename(diagnosis) if diagnosis else "random"
    output_file = OUTPUT_DIR / f"{file_index:04d}_dialogue_{diagnosis_slug}_skill{doctor_skill}.mp3"

    if diagnosis:
        print(f"Generating new dialogue for diagnosis: {diagnosis} (doctor skill: {doctor_skill}/10)...")
    else:
        print(f"Generating new dialogue (doctor skill: {doctor_skill}/10)...")
    
    system_prompt = get_dialogue_generation_prompt(diagnosis=diagnosis, doctor_skill=doctor_skill)
    generated_dialogue = await llm.generate_dialogue(system_prompt=system_prompt, diagnosis=diagnosis)
    dialogue_script = generated_dialogue.dialogue

    print(f"Generating dialogue audio to {output_file}...")

    temp_files: list[Path] = []

    try:
        # Generate audio for each turn
        for idx, turn in enumerate(dialogue_script):
            role = turn.role
            voice = turn.voice
            text = turn.text
            print(f"[{idx+1}/{len(dialogue_script)}] {role}: {text[:30]}...")

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
        with open(output_file, "wb") as outfile:
            for fpath in temp_files:
                with open(fpath, "rb") as infile:
                    outfile.write(infile.read())

        print(f"Done! Saved to {output_file}")

    except Exception as e:
        print(f"Error generating audio: {e}")
    finally:
        # Cleanup temp files
        print("Cleaning up temporary files...")
        for fpath in temp_files:
            if fpath.exists():
                fpath.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample dialogue audio")
    parser.add_argument(
        "--diagnosis",
        type=str,
        default=None,
        help="Specific diagnosis to generate dialogue about (optional)",
    )
    parser.add_argument(
        "--doctor-skill",
        type=int,
        default=5,
        choices=range(0, 11),
        metavar="[0-10]",
        help="Doctor's skill level from 0 (complete novice) to 10 (expert master). Default: 5 (competent with 2 years experience)",
    )
    args = parser.parse_args()
    
    asyncio.run(generate_dialogue_audio(diagnosis=args.diagnosis, doctor_skill=args.doctor_skill))
