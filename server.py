import os
import subprocess
import tempfile
import threading
import uuid
import sqlite3
import time
import requests
import anthropic
from flask import Flask, request, jsonify, render_template, abort

app = Flask(__name__)

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DB_PATH = "/tmp/jobs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT,
            summary TEXT,
            error TEXT
        )
    """)
    conn.commit()
    conn.close()

def set_job(job_id, status, summary=None, error=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO jobs (id, status, summary, error) VALUES (?, ?, ?, ?)",
                 (job_id, status, summary, error))
    conn.commit()
    conn.close()

def get_job(job_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT status, summary, error FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {"status": row[0], "summary": row[1], "error": row[2]}

def get_metadata(url):
    cmd = ["yt-dlp", "--dump-json", "--no-download", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    import json
    data = json.loads(result.stdout)
    return {
        "title": data.get("title", ""),
        "description": data.get("description", ""),
        "uploader": data.get("uploader", ""),
    }

def download_audio(url, output_path):
    cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "--no-check-formats", "-o", output_path, url]
    subprocess.run(cmd, check=True)

def transcribe(audio_path):
    aai_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not aai_key:
        raise Exception("ASSEMBLYAI_API_KEY not set")
    headers = {"authorization": aai_key}

    print("Uploading audio to AssemblyAI...")
    with open(audio_path, "rb") as f:
        upload = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
    print(f"Upload response: {upload.status_code} {upload.text[:200]}")
    audio_url = upload.json()["upload_url"]

    response = requests.post("https://api.assemblyai.com/v2/transcript",
                             headers=headers,
                             json={"audio_url": audio_url, "speech_models": ["universal-2"]})
    print(f"Transcript request: {response.status_code} {response.text[:200]}")
    transcript_id = response.json()["id"]

    print(f"Polling transcript {transcript_id}...")
    while True:
        result = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                              headers=headers).json()
        print(f"Status: {result['status']}")
        if result["status"] == "completed":
            return result["text"]
        elif result["status"] == "error":
            raise Exception(f"AssemblyAI error: {result['error']}")
        time.sleep(3)

def summarize(transcript, metadata={}):
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

Video metadata:
Title: {metadata.get('title', 'N/A')}
Creator: {metadata.get('uploader', 'N/A')}
Description: {metadata.get('description', 'N/A')}

Transcript:
{transcript}"""}]
    )
    return message.content[0].text

def process_job(job_id, url):
    try:
        set_job(job_id, "processing")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")

            metadata_result = {}
            def fetch_metadata():
                metadata_result.update(get_metadata(url))

            meta_thread = threading.Thread(target=fetch_metadata)
            meta_thread.start()
            download_audio(url, audio_path)
            meta_thread.join()

            transcript = transcribe(audio_path)
            summary = summarize(transcript, metadata_result)
            note_content = f"Source: {url}\n\n{summary}"
            set_job(job_id, "done", summary=note_content)
    except Exception as e:
        set_job(job_id, "error", error=str(e))

@app.route("/summarize", methods=["POST"])
def handle_summarize():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    job_id = str(uuid.uuid4())
    set_job(job_id, "queued")
    thread = threading.Thread(target=process_job, args=(job_id, url))
    thread.start()
    return jsonify({"job_id": job_id})

@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)

@app.route("/text/<job_id>", methods=["GET"])
def get_text(job_id):
    job = get_job(job_id)
    if not job or job["status"] != "done":
        return "", 204
    return job["summary"], 200, {"Content-Type": "text/plain"}

CONTRACT_PASSWORD = os.environ.get("CONTRACT_PASSWORD", "contrato123")

@app.route("/contract")
def contract():
    pwd = request.args.get("pwd", "")
    if pwd != CONTRACT_PASSWORD:
        return """
        <html><body style="font-family:sans-serif;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;margin:0">
        <h2>Enter password</h2>
        <form method="get">
            <input name="pwd" type="password" placeholder="Password" style="padding:10px;font-size:16px;border:1px solid #ccc;border-radius:6px" autofocus>
            <button type="submit" style="margin-left:8px;padding:10px 20px;font-size:16px;background:#2563eb;color:white;border:none;border-radius:6px;cursor:pointer">View</button>
        </form>
        </body></html>
        """, 401
    return render_template("contract.html")


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
