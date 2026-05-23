import os
import subprocess
import tempfile
import threading
import uuid
import whisper
import anthropic
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

jobs = {}

def download_audio(url, output_path):
    cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_path, url]
    subprocess.run(cmd, check=True)

def transcribe(audio_path):
    model = whisper.load_model("tiny")
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

def process_job(job_id, url):
    try:
        jobs[job_id] = {"status": "processing"}
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")
            download_audio(url, audio_path)
            transcript = transcribe(audio_path)
            summary = summarize(transcript)
            jobs[job_id] = {"status": "done", "summary": summary}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}

@app.route("/summarize", methods=["POST"])
def handle_summarize():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    job_id = str(uuid.uuid4())
    thread = threading.Thread(target=process_job, args=(job_id, url))
    thread.start()
    return jsonify({"job_id": job_id})

@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
