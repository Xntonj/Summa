import subprocess
import sys

url = input("Paste video URL: ").strip()
subprocess.run([sys.executable, "/Users/anaranjo/video-summarizer/summarize.py", url])
