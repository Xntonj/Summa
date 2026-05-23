import sys
import os
import subprocess
import tempfile
import whisper
import anthropic

API_KEY = "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"

def download_audio(url, output_path):
    cmd = [
        os.path.expanduser("~/Library/Python/3.9/bin/yt-dlp"),
        "-x", "--audio-format", "mp3",
        "--ffmpeg-location", os.path.expanduser("~/bin/ffmpeg"),
        "-o", output_path,
        url
    ]
    subprocess.run(cmd, check=True)

def transcribe(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def summarize(transcript):
    client = anthropic.Anthropic(api_key=API_KEY)
    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are summarizing a video transcript. First, detect which category best fits the content:

1. Self-improvement/tips — Steps or advice on how to improve your life
2. Neurodivergence — Explaining what it is, how it shows up, how to cope
3. Wellness/manifestation — Spiritual practices, energy, abundance, pain management
4. News/debates — Current events, politics, debates between people

Then summarize accordingly:
- Self-improvement: Extract the main tips as clear actionable steps
- Neurodivergence: Keep the explanation, real-life examples, and coping strategies
- Wellness/manifestation: Keep the concepts and practical steps, no judgment on whether it's scientific
- News/debates: Clarify the topic, what each side is saying, and why it matters

Use your judgment — keep what's relevant, skip the fluff. Don't be too rigid about structure. Start with the detected category label.

Transcript:
{transcript}"""}]
    )
    return message.content[0].text

if __name__ == "__main__":
    url = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        print("Downloading audio...")
        download_audio(url, audio_path)
        print("Transcribing...")
        transcript = transcribe(audio_path)
        print("Summarizing...")
        summary = summarize(transcript)
        print(summary)
