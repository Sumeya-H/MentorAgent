import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=150,
    separators=["\n## ","\n### ","\n\n","\n"," "]
)

LEVEL_RE = re.compile(r"(Level\s+\d+[:\-]?\s*[A-Za-z0-9 \-]*)", re.IGNORECASE)
SECTION_RE = re.compile(r"(Section\s+\d+[:\-]?\s*[A-Za-z0-9 \-]*)", re.IGNORECASE)

def enrich_metadata(doc: Document, idx: int) -> Document:
    text = doc.page_content[:600]
    meta = dict(doc.metadata) if doc.metadata else {}
    le = LEVEL_RE.search(text)
    se = SECTION_RE.search(text)
    if le: meta["level"] = le.group(1).strip()
    if se: meta["section"] = se.group(1).strip()
    meta["chunk_id"] = idx
    meta.setdefault("source_file", meta.get("repo") or meta.get("video_id") or meta.get("path") or "unknown")
    meta["reference"] = f"{meta.get('level','Level ?')}, {meta.get('section','Section ?')}"
    return Document(page_content=doc.page_content, metadata=meta)

def split_and_tag(docs: list[Document]) -> list[Document]:
    chunks = []
    for d in docs:
        parts = SPLITTER.split_documents([d])
        for i, p in enumerate(parts):
            chunks.append(enrich_metadata(p, i))
    return chunks
