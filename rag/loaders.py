import os
import re
import yt_dlp
from typing import List
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, NotebookLoader
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
import base64, requests, json

def load_pdf(path: Path) -> list[Document]:
    return PyPDFLoader(str(path)).load()

def load_markdown(path: Path) -> list[Document]:
    return TextLoader(str(path), encoding="utf-8").load()

def load_ipynb(path: Path) -> list[Document]:
    # Prefer LangChain’s NotebookLoader; fallback if needed
    try:
        return NotebookLoader(str(path)).load()
    except Exception:
        data = json.loads(path.read_text(encoding="utf-8"))
        cells = data.get("cells", [])
        text = "\n".join("".join(c.get("source", [])) for c in cells)
        return [Document(page_content=text, metadata={"source_file": path.name, "type": "ipynb"})]

#def load_youtube_transcript(url_or_id: str) -> list[Document]:
#    vid = url_or_id.split("v=")[-1].split("&")[0] if "youtube" in url_or_id else url_or_id
#    transcript = YouTubeTranscriptApi.fetch([vid], languages=["en"])
#    text = " ".join([t.text for t in transcript])
#    return [Document(page_content=text, metadata={"source": "youtube", "video_id": vid, "type":"youtube"})]

def _parse_transcript(raw_text: str) -> str:
    """
    Parses and cleans the raw transcript output from yt-dlp.
    Removes timestamps, metadata, and duplicate lines.
    """

    # 1. Remove XML-like tags, e.g., <c> </c>
    text = re.sub(r'<[^>]+>', '', raw_text)
    # 2. Remove bracketed sound descriptions, e.g., [ CHEERS AND APPLAUSE ]
    text = re.sub(r'\[.*?\]', '', text)
    # 3. Remove metadata headers like "Kind: captions Language: en"
    text = re.sub(r'^Kind:.*?\n', '', text, flags=re.MULTILINE)
    # 4. Remove WEBVTT header if present
    text = re.sub(r'^WEBVTT.*\n', '', text, flags=re.MULTILINE)
    # 5. Remove timestamp lines in VTT format
    text = re.sub(
        r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n',
        '',
        text
    )
    # 6. Split into lines, clean up, and remove duplicates
    lines = text.strip().split('\n')
    cleaned_lines = []
    for line in lines:
        clean_line = line.strip().lstrip('> ').strip()
        if clean_line and (not cleaned_lines or cleaned_lines[-1] != clean_line):
            cleaned_lines.append(clean_line)
    # 7. Join and normalize whitespace
    final_text = " ".join(cleaned_lines)
    return re.sub(r'\s+', ' ', final_text).strip()

def load_youtube_transcript(youtube_url: str) -> List[Document]:
    """
    Downloads a transcript from YouTube using yt-dlp, parses it, and returns it
    as a list containing a single LangChain Document.
    """
    # Use a temporary file name template
    temp_filename_template = "temp_transcript_%(id)s"

    # We ask for the 'live_chat' format when available as it often has a 
    # cleaner text structure, but we will clean it regardless.
    ydl_opts = {
        'format': 'best',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt', # Request VTT format specifically
        'skip_download': True,
        'outtmpl': temp_filename_template,
        'quiet': True,
        'ignoreerrors': True,
    }

    transcript_text = ""
    downloaded_file = None
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_id = info_dict.get('id', 'default')
            
            # Construct the expected filename
            expected_filename = f"temp_transcript_{video_id}.en.vtt"
            
            # Trigger the download
            ydl.download([youtube_url])

            if os.path.exists(expected_filename):
                downloaded_file = expected_filename
                with open(downloaded_file, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                transcript_text = _parse_transcript(raw_content)
            else:
                raise FileNotFoundError("yt-dlp failed to download the transcript file in VTT format.")

    except Exception as e:
        raise RuntimeError(f"Failed to process youtube video {youtube_url}: {e}")
    finally:
        # Clean up the downloaded file
        if downloaded_file and os.path.exists(downloaded_file):
            os.remove(downloaded_file)

    if not transcript_text:
        raise ValueError("Could not extract any text from the transcript.")

    doc = Document(
        page_content=transcript_text,
        metadata={"source": "youtube", "video_id": youtube_url, "type":"youtube"}
    )

    return [doc]

def load_github_readme(owner_repo: str) -> list[Document]:
    api = f"https://api.github.com/repos/{owner_repo}/readme"
    r = requests.get(api, headers={"Accept":"application/vnd.github+json"})
    if r.status_code != 200: 
        return []
    content = base64.b64decode(r.json()["content"]).decode("utf-8", errors="ignore")
    return [Document(page_content=content, metadata={"source":"github", "repo":owner_repo, "path":"README.md","type":"readme"})]
