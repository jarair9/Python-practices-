import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import whisper
from pytubefix import YouTube
# import streamlit as st
import edge_tts

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("Environment variable GEMINI_API_KEY not found")
genai.configure(api_key=GEMINI_API_KEY)



def downloader(url):
    try:
        yt = YouTube(url, use_oauth=False, allow_oauth_cache=True)

        stream = (
            yt.streams
            .filter(progressive=True, file_extension='mp4')
            .order_by('resolution')
            .first()
        )

        if stream is None:
            print("No downloadable stream found.")
            return False

        stream.download(filename="video.mp4")
        print("The video downloaded successfully!")
        return True

    except Exception as e:
        print("An error occurred:", e)
        return False


def transcribing(video: str = "video.mp4") -> str:
    print("Loading Whisper model (this may take a moment)...")
    model = whisper.load_model("small")
    print(f"Transcribing {video} ...")
    result = model.transcribe(video)

    if not os.path.exists("text_files"):
        os.makedirs("text_files")

    out_path = "text_files/video_trans.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Transcription saved to: {out_path}")
    return result["text"]


def jallm(text_file: str = "text_files/video_trans.txt"):
    if not os.path.exists(text_file):
        print(f"Transcription file not found: {text_file}")
        return

    with open(text_file, "r", encoding="utf-8") as f:
        transcription_text = f.read().strip()

    # Improved system prompt (keeps model strictly on transcription unless user asks otherwise)
    system_prompt = f"""
    You are a script and transcription analysis assistant.
    Your role is to carefully analyze and understand the provided transcription.
    After the user asks a question, answer strictly based on the information contained in that transcription, similar to how NotebookLM works.

    Do not use outside knowledge unless the user explicitly asks for it.
    If the transcription does not contain enough information to answer a question, say so clearly and ask the user to clarify or provide more context.

    The transcription is:
    {transcription_text}
"""

    # Create Gemini model with system instruction
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=system_prompt,
    )

    chat = model.start_chat(history=[])

    print("\nChat ready. Type questions about the transcription. Type 'exit' to quit.\n")

    while True:
        try:
            msg = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not msg:
            continue

        if msg.lower() == "exit":
            print("Goodbye.")
            break

        try:
            response = chat.send_message(msg)
            # Some SDKs return .text, others .response.text(); handle gracefully:
            ai_text = getattr(response, "text", None) or getattr(response, "response", None)
            if isinstance(ai_text, dict) and "text" in ai_text:
                ai_text = ai_text["text"]
            if ai_text is None:
                # fallback
                ai_text = str(response)
            print("\nAI:", ai_text, "\n")

        except Exception as e:
            print("Error while calling Gemini:", str(e))
            # Optional: retry or continue
            time.sleep(1)


def main():
    print("Welcome to JA Studio LLM Video talker")
    print("Just drop a YouTube URL and talk with the video .\n")
    print()

    url = input("Enter YouTube video URL: ").strip()
    if not url:
        print("No URL provided. Exiting.")
        return

    ok = downloader(url)
    if not ok:
        print("Download failed. Exiting.")
        return

    transcribing()
    jallm()


if __name__ == "__main__":
    main()
'https://youtube.be/nIfwCG2rqlg?si=gfieNP_EnUdOIxf6'