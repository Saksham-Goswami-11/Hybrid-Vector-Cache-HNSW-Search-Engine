import os
import hashlib
import time
from typing import List, Dict
import tiktoken
from pypdf import PdfReader
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI
from client import SynapseClient

# Setup OpenAI client only if configured
openai_client = None
if os.environ.get("OPENAI_API_KEY"):
    openai_client = OpenAI()

NEARBY_PORT = int(os.environ.get("NEARBY_PORT", 6380))
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "./knowledge_base")

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            return text
        except Exception as e:
            print(f"Error parsing PDF {filepath}: {e}")
            return ""
    elif ext in ['.md', '.txt']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file {filepath}: {e}")
            return ""
    return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks

class IngestionDaemon:
    def __init__(self, directory: str = KNOWLEDGE_BASE_DIR, port: int = NEARBY_PORT):
        self.directory = os.path.abspath(directory)
        self.port = port
        self.client = SynapseClient(port=self.port)
        self.processed_files: Dict[str, str] = {} # filepath -> file_hash
        
        # Ensure knowledge base directory exists
        os.makedirs(self.directory, exist_ok=True)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if not openai_client:
            # Fallback to deterministic mocks for offline dev
            mock_embeddings = []
            for text in texts:
                h = hashlib.sha256(text.encode()).digest()
                vec = []
                for idx in range(1536):
                    # Deterministic values [0, 1] based on string hash
                    val = float(h[idx % 32]) / 256.0
                    vec.append(val)
                mock_embeddings.append(vec)
            return mock_embeddings

        # Fetch real embeddings
        resp = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in resp.data]

    def ingest_file(self, filepath: str):
        if not os.path.exists(filepath):
            return
        
        # Prevent symlink traversal
        if os.path.islink(filepath):
            print(f"SECURITY WARNING: Ignoring symlink {filepath}")
            return
            
        # Jail path resolution
        try:
            resolved_target = os.path.realpath(filepath)
            resolved_base = os.path.realpath(self.directory)
            
            # Ensure file lives strictly inside base directory
            if not resolved_target.startswith(resolved_base + os.sep) and resolved_target != resolved_base:
                print(f"SECURITY WARNING: Path traversal detected. {resolved_target} escapes {resolved_base}")
                return
        except Exception as e:
            print(f"Error resolving path {filepath}: {e}")
            return

        # Skip directories
        if os.path.isdir(filepath):
            return

        # Skip dotfiles/hidden files
        if os.path.basename(filepath).startswith('.'):
            return

        # Validate file type
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ['.md', '.txt', '.pdf']:
            return

        file_hash = get_file_hash(filepath)
        if self.processed_files.get(filepath) == file_hash:
            # File hasn't changed, skip ingestion
            return

        print(f"Ingesting: {os.path.basename(filepath)}")
        text = extract_text(filepath)
        if not text.strip():
            print(f"Skipping empty file: {filepath}")
            return

        chunks = chunk_text(text)
        if not chunks:
            return

        try:
            embeddings = self.generate_embeddings(chunks)
        except Exception as e:
            print(f"Failed to generate embeddings for {filepath}: {e}")
            return

        filename = os.path.basename(filepath)
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_key = f"chunk:{filename}:{idx}"
            
            # Store original text payload
            self.client.set(chunk_key, chunk)
            
            # Store dense embedding
            self.client.vset(
                namespace="docs",
                key=chunk_key,
                vector=vector,
                metadata={"source": filename}
            )

        self.processed_files[filepath] = file_hash
        print(f"Successfully ingested {filepath} ({len(chunks)} chunks)")

    def scan_directory(self):
        print(f"Scanning directory: {self.directory}")
        for root, _, files in os.walk(self.directory):
            for file in files:
                filepath = os.path.join(root, file)
                self.ingest_file(filepath)

    def start(self):
        # Run initial scan
        self.scan_directory()

        event_handler = IngestionHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.directory, recursive=True)
        observer.start()
        print(f"Watchdog Ingestion Daemon started on directory: {self.directory}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

class IngestionHandler(FileSystemEventHandler):
    def __init__(self, daemon: IngestionDaemon):
        self.daemon = daemon

    def on_created(self, event):
        if not event.is_directory:
            self.daemon.ingest_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.daemon.ingest_file(event.src_path)

if __name__ == "__main__":
    print("Starting watchdog ingestion daemon...")
    daemon = IngestionDaemon()
    daemon.start()
