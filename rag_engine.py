import os
import math
from typing import List, Dict, Any, Tuple, Optional

# Attempt numpy import with pure-python fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Attempt google-generativeai import
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

# Model constants
DEFAULT_GENERATION_MODEL = "gemini-3.8-flash"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


def list_generation_models(api_key: str) -> List[str]:
    """Return Gemini models available to this API key that support generateContent."""
    if not HAS_GENAI or not api_key:
        return []
    genai.configure(api_key=api_key)
    names = [
        m.name.removeprefix("models/")
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods and "gemini" in m.name
    ]
    return sorted(names, reverse=True)


class DocumentChunk:
    """Represents a chunk of a document stored in memory."""
    def __init__(self, chunk_id: int, source: str, text: str, embedding: Optional[List[float]] = None):
        self.chunk_id = chunk_id
        self.source = source
        self.text = text
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "has_embedding": self.embedding is not None
        }


def _pure_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Pure-python fallback for cosine similarity."""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


class MemoryVectorStore:
    """In-memory vector store using cosine similarity."""
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self._matrix: Optional[Any] = None

    def clear(self):
        self.chunks = []
        self._matrix = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        self.chunks.extend(chunks)
        valid_embeddings = [c.embedding for c in self.chunks if c.embedding is not None]
        if valid_embeddings and HAS_NUMPY:
            self._matrix = np.array(valid_embeddings, dtype=np.float32)
        else:
            self._matrix = None

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Search for top_k most similar chunks using cosine similarity."""
        if not self.chunks:
            return []

        # Fast path with NumPy
        if HAS_NUMPY and self._matrix is not None and len(self._matrix) > 0:
            query_vec = np.array(query_embedding, dtype=np.float32)
            doc_norms = np.linalg.norm(self._matrix, axis=1)
            query_norm = np.linalg.norm(query_vec)

            if query_norm == 0:
                return []

            doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
            similarities = np.dot(self._matrix, query_vec) / (doc_norms * query_norm)

            k = min(top_k, len(self.chunks))
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                results.append((self.chunks[idx], score))
            return results

        # Pure Python fallback
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                score = _pure_cosine_similarity(query_embedding, chunk.embedding)
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def __len__(self):
        return len(self.chunks)


def load_file_content(filepath: str) -> Optional[str]:
    """Read a document file with multiple encoding fallbacks."""
    encodings = ["utf-8", "utf-8-sig", "cp874", "tis-620", "latin-1"]
    
    # Handle PDF if pypdf is installed
    if filepath.lower().endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            text_parts = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(text_parts).strip()
        except ImportError:
            print(f"[Warning] pypdf not installed. Skipping PDF file: {filepath}")
            return None
        except Exception as e:
            print(f"[Error] Failed reading PDF {filepath}: {e}")
            return None

    # Text / Markdown files
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
        except Exception as e:
            print(f"[Error] Failed reading {filepath} with {enc}: {e}")
            return None
    return None


def split_text_into_chunks(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into chunks intelligently, respecting paragraph and line breaks.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Split by double newlines first (paragraphs)
    paragraphs = text.split("\n\n")
    raw_pieces = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) <= chunk_size:
            raw_pieces.append(p)
        else:
            # Split by single newline
            lines = p.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if len(line) <= chunk_size:
                    raw_pieces.append(line)
                else:
                    # Split by character with word boundary consideration
                    start = 0
                    while start < len(line):
                        end = min(start + chunk_size, len(line))
                        if end < len(line):
                            last_space = line.rfind(" ", start, end)
                            if last_space != -1 and last_space > start + chunk_size // 2:
                                end = last_space
                        raw_pieces.append(line[start:end].strip())
                        start = end

    # Combine small pieces into target chunk_size with overlap
    chunks = []
    current_chunk = ""

    for piece in raw_pieces:
        if not piece:
            continue
        if not current_chunk:
            current_chunk = piece
        elif len(current_chunk) + len(piece) + 2 <= chunk_size:
            current_chunk += "\n\n" + piece
        else:
            chunks.append(current_chunk)
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                overlap_text = current_chunk[-chunk_overlap:]
                space_idx = overlap_text.find(" ")
                if space_idx != -1 and space_idx < len(overlap_text) // 2:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk = overlap_text + "\n\n" + piece
            else:
                current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class GeminiRAG:
    """Main RAG engine handling document processing, embedding, and generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        generation_model: str = DEFAULT_GENERATION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ):
        if not HAS_GENAI:
            raise ImportError(
                "แพ็กเกจ 'google-generativeai' ยังไม่ได้ติดตั้ง กรุณาติดตั้งด้วยคำสั่ง: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.generation_model_name = generation_model
        self.embedding_model_name = embedding_model
        self.vector_store = MemoryVectorStore()
        self.indexed_files: List[str] = []

        if self.api_key:
            genai.configure(api_key=self.api_key)

    def set_api_key(self, api_key: str):
        """Update API key and reconfigure genai."""
        self.api_key = api_key
        if HAS_GENAI and self.api_key:
            genai.configure(api_key=self.api_key)

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Compute embedding for a single text using Gemini Embedding API."""
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured.")

        response = genai.embed_content(
            model=self.embedding_model_name,
            content=text,
            task_type=task_type
        )
        return response["embedding"]

    def get_embeddings_batch(
        self, texts: List[str], task_type: str = "retrieval_document", batch_size: int = 50
    ) -> List[List[float]]:
        """Compute embeddings in batches to stay within API limits."""
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured.")

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = genai.embed_content(
                model=self.embedding_model_name,
                content=batch,
                task_type=task_type
            )
            embeddings = response.get("embedding", [])
            all_embeddings.extend(embeddings)

        return all_embeddings

    def index_directory(
        self,
        docs_dir: str = "docs",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        progress_callback=None
    ) -> int:
        """
        Scan docs directory, split files into chunks, compute embeddings,
        and store in memory vector store.
        """
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)
            return 0

        self.vector_store.clear()
        self.indexed_files = []

        supported_exts = {".txt", ".md", ".pdf"}
        filenames = sorted([
            f for f in os.listdir(docs_dir)
            if os.path.splitext(f)[1].lower() in supported_exts and not f.startswith(".")
        ])

        if not filenames:
            return 0

        all_raw_chunks: List[Tuple[str, str]] = []

        for filename in filenames:
            filepath = os.path.join(docs_dir, filename)
            content = load_file_content(filepath)
            if not content:
                continue

            chunks = split_text_into_chunks(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for c in chunks:
                all_raw_chunks.append((filename, c))

            self.indexed_files.append(filename)

        if not all_raw_chunks:
            return 0

        texts = [chunk_text for _, chunk_text in all_raw_chunks]
        
        batch_size = 50
        all_embeddings: List[List[float]] = []
        total_batches = math.ceil(len(texts) / batch_size)

        for b_idx in range(total_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            batch_embeds = self.get_embeddings_batch(batch_texts, task_type="retrieval_document")
            all_embeddings.extend(batch_embeds)

            if progress_callback:
                progress_callback(min(1.0, (b_idx + 1) / total_batches))

        doc_chunks = []
        for idx, ((source, text), emb) in enumerate(zip(all_raw_chunks, all_embeddings)):
            doc_chunks.append(DocumentChunk(chunk_id=idx, source=source, text=text, embedding=emb))

        self.vector_store.add_chunks(doc_chunks)
        return len(doc_chunks)

    def search_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Embed user query and search for top_k most similar chunks."""
        query_embedding = self.get_embedding(query, task_type="retrieval_query")
        return self.vector_store.search(query_embedding, top_k=top_k)

    def generate_prompt(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Construct the prompt with retrieved context."""
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[แหล่งข้อมูล {i}: {chunk.source} (ความเกี่ยวข้อง: {score*100:.1f}%)]\n{chunk.text}"
            )

        context_str = "\n\n---\n\n".join(context_parts) if context_parts else "ไม่มีข้อมูลที่เกี่ยวข้องในเอกสาร"

        prompt = f"""คุณคือผู้ช่วย AI อัจฉริยะที่ตอบคำถามอย่างถูกต้อง แม่นยำ และสุภาพ
กรุณาตอบคำถามของผู้ใช้โดยอ้างอิงจากข้อมูลบริบท (Context) ที่กำหนดให้ด้านล่างนี้เท่านั้น:

=== ข้อมูลบริบท (CONTEXT) ===
{context_str}
=============================

คำแนะนำสำคัญ:
1. ตอบคำถามโดยใช้ข้อเท็จจริงที่มีใน Context ด้านบนเท่านั้น
2. หากใน Context ไม่มีข้อมูลเพียงพอที่จะตอบคำถามได้อย่างชัดเจน ให้ตอบอย่างสุภาพว่า "ขออภัยครับ/ค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่จัดเตรียมไว้" ห้ามแต่งหรือคาดเดาข้อมูลขึ้นมาเอง (Zero Hallucination)
3. ระบุชื่อแหล่งข้อมูล (เอกสารอ้างอิง) ในคำตอบเมื่อเหมาะสม เพื่อให้ผู้ใช้อ้างอิงได้
4. จัดรูปแบบคำตอบด้วย Markdown ให้อ่านง่าย เป็นระเบียบ มีหัวข้อหรือ bullet point เมื่อเหมาะสม

คำถามของผู้ใช้: {query}

คำตอบ:"""
        return prompt

    def answer_query_stream(self, query: str, top_k: int = 3):
        """
        Perform RAG retrieval and stream Gemini LLM generation.
        Yields (response_stream, retrieved_chunks).
        """
        retrieved_chunks = self.search_relevant_chunks(query, top_k=top_k)
        prompt = self.generate_prompt(query, retrieved_chunks)

        model = genai.GenerativeModel(self.generation_model_name)
        response_stream = model.generate_content(prompt, stream=True)

        return response_stream, retrieved_chunks
