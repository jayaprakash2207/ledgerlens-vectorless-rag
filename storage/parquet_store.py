import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class StoredDocument:
    doc_id: str
    source_name: str
    file_type: str
    created_at: str
    text_len: int


class ParquetStore:
    def __init__(self, base_dir: str = "data/store") -> None:
        self.base_dir = base_dir
        self.docs_path = os.path.join(self.base_dir, "documents.parquet")
        self.chunks_path = os.path.join(self.base_dir, "chunks.parquet")
        os.makedirs(self.base_dir, exist_ok=True)

    def save_document(
        self,
        source_name: str,
        file_type: str,
        raw_text: str,
        sections: Dict[str, str],
    ) -> StoredDocument:
        doc_id = self._doc_id(source_name, raw_text)
        created_at = datetime.now(timezone.utc).isoformat()

        doc_row = {
            "doc_id": doc_id,
            "source_name": source_name,
            "file_type": file_type,
            "created_at": created_at,
            "text_len": len(raw_text),
            "raw_text": raw_text,
        }
        self._append_rows(self.docs_path, [doc_row], dedupe_key="doc_id")

        chunk_rows = []
        for section, text in sections.items():
            if not text:
                continue
            for index, chunk in enumerate(self._chunk_text(text)):
                chunk_rows.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}:{section}:{index}",
                        "section": section,
                        "chunk_index": index,
                        "text": chunk,
                    }
                )

        if chunk_rows:
            self._append_rows(self.chunks_path, chunk_rows, dedupe_key="chunk_id")

        return StoredDocument(
            doc_id=doc_id,
            source_name=source_name,
            file_type=file_type,
            created_at=created_at,
            text_len=len(raw_text),
        )

    def load_chunks(self, doc_id: str) -> List[Dict[str, object]]:
        if not os.path.exists(self.chunks_path):
            return []
        df = pd.read_parquet(self.chunks_path)
        subset = df[df["doc_id"] == doc_id]
        return subset.to_dict(orient="records")

    def list_documents(self) -> List[Dict[str, object]]:
        if not os.path.exists(self.docs_path):
            return []
        df = pd.read_parquet(self.docs_path)
        df = df.sort_values(by="created_at", ascending=False)
        return df.to_dict(orient="records")

    def load_document(self, doc_id: str) -> Optional[Dict[str, object]]:
        if not os.path.exists(self.docs_path):
            return None
        df = pd.read_parquet(self.docs_path)
        subset = df[df["doc_id"] == doc_id]
        if subset.empty:
            return None
        return subset.iloc[0].to_dict()

    def _append_rows(self, path: str, rows: List[Dict[str, object]], dedupe_key: str) -> None:
        new_df = pd.DataFrame(rows)
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[dedupe_key], keep="last")
            combined.to_parquet(path, index=False)
        else:
            new_df.to_parquet(path, index=False)

    def _chunk_text(self, text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
        sentences = [part.strip() for part in text.split(". ") if part.strip()]
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            sentence_len = len(sentence) + 2
            if current_len + sentence_len > max_chars and current:
                chunk = ". ".join(current).strip()
                chunks.append(chunk)
                overlap_text = chunk[-overlap:] if overlap < len(chunk) else chunk
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current.append(sentence)
            current_len += sentence_len

        if current:
            chunks.append(". ".join(current).strip())

        return chunks

    def _doc_id(self, source_name: str, raw_text: str) -> str:
        digest = hashlib.sha256((source_name + raw_text[:5000]).encode("utf-8", errors="ignore")).hexdigest()
        return digest[:16]
