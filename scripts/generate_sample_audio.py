import asyncio
import os
import sys
from pathlib import Path

# Add project root to path to allow imports from app
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.services.llm import get_llm_provider
from config.prompts import SYSTEM_PROMPT_GENERATE_DIALOGUE
from config.settings import config

# Load environment variables (reusing .env handling logic minimally for this script)
load_dotenv()
API_KEY = config.OPENAI_API_KEY

if not API_KEY:
    print("Error: OPENAI_API_KEY not found in .env")
    exit(1)


OUTPUT_DIR = config.TEMP_DIR


def get_next_file_index() -> int:
    existing_files = list(OUTPUT_DIR.glob("*_dialogue_sample.mp3"))
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


async def generate_dialogue_audio() -> None:
    client = AsyncOpenAI(api_key=API_KEY)
    llm = get_llm_provider()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    file_index = get_next_file_index()
    output_file = OUTPUT_DIR / f"{file_index:04d}_dialogue_sample.mp3"

    print("Generating new dialogue...")
    generated_dialogue = await llm.generate_dialogue(SYSTEM_PROMPT_GENERATE_DIALOGUE)
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
    asyncio.run(generate_dialogue_audio())
