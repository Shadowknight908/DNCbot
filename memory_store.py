"""On-device vector store for roleplay memories.

v0.6 changes:
- get_by_author_before(): metadata-filtered retrieval for GM rulings
  (find an author's prior posts before a given date)
- Memory metadata now stores full message text, not just an excerpt
- Supports entry_type='ruling' for storing GM rulings as memories
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(
        self,
        db_path: str,
        collection_name: str = "lore",
        tombstone_path: Optional[str] = None,
    ):
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._tombstone_path = tombstone_path or os.path.join(
            db_path, "voided_memories.jsonl"
        )
        os.makedirs(os.path.dirname(self._tombstone_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Add / search
    # ------------------------------------------------------------------
    def add(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        memory_id = str(uuid.uuid4())
        meta: Dict[str, Any] = {"stored_at": _now_iso()}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is None:
                    meta[k] = ""
                else:
                    meta[k] = str(v)
        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
        return memory_id

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        ids = results.get("ids", [[]])[0]
        if not ids:
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        return [
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i] or {},
                "distance": dists[i],
            }
            for i in range(len(ids))
        ]

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def has_source_message(self, source_message_id: str) -> bool:
        try:
            result = self._collection.get(
                where={"source_message_id": str(source_message_id)},
                limit=1,
            )
            return bool(result.get("ids"))
        except Exception:
            return False

    def get_by_source_message(
        self, source_message_id: str
    ) -> Optional[Dict[str, Any]]:
        result = self._collection.get(
            where={"source_message_id": str(source_message_id)},
            include=["documents", "metadatas", "embeddings"],
            limit=1,
        )
        if not result.get("ids"):
            return None
        return {
            "id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0] or {},
            "embedding": result["embeddings"][0],
        }

    def get_all_by_source_message(
        self, source_message_id: str
    ) -> List[Dict[str, Any]]:
        result = self._collection.get(
            where={"source_message_id": str(source_message_id)},
            include=["documents", "metadatas", "embeddings"],
        )
        if not result.get("ids"):
            return []
        return [
            {
                "id": result["ids"][i],
                "text": result["documents"][i],
                "metadata": result["metadatas"][i] or {},
                "embedding": result["embeddings"][i],
            }
            for i in range(len(result["ids"]))
        ]

    def get_rulings_by_target_message(
        self, ruled_on_message_id: str
    ) -> List[Dict[str, Any]]:
        result = self._collection.get(
            where={"ruled_on_message_id": str(ruled_on_message_id)},
            include=["documents", "metadatas", "embeddings"],
        )
        if not result.get("ids"):
            return []
        return [
            {
                "id": result["ids"][i],
                "text": result["documents"][i],
                "metadata": result["metadatas"][i] or {},
                "embedding": result["embeddings"][i],
            }
            for i in range(len(result["ids"]))
        ]

    def get_by_author(self, author_id: str) -> List[Dict[str, Any]]:
        result = self._collection.get(
            where={"author_id": str(author_id)},
            include=["documents", "metadatas", "embeddings"],
        )
        if not result.get("ids"):
            return []
        return [
            {
                "id": result["ids"][i],
                "text": result["documents"][i],
                "metadata": result["metadatas"][i] or {},
                "embedding": result["embeddings"][i],
            }
            for i in range(len(result["ids"]))
        ]

    def get_by_author_before(
        self, author_id: str, before_iso: str
    ) -> List[Dict[str, Any]]:
        """All memories by an author with a timestamp strictly before a given ISO datetime.
        Used by GM mode to find an author's prior posts when ruling on a new one.
        """
        # ChromaDB's where clause supports $and + $lt for ISO strings (lexicographic
        # comparison is correct for ISO-8601). But we filter in Python because Chroma's
        # operator support varies by version — small N anyway.
        memories = self.get_by_author(author_id)
        return [
            m for m in memories
            if (m["metadata"].get("timestamp") or "") < before_iso
        ]

    def get_all(self) -> List[Dict[str, Any]]:
        result = self._collection.get(include=["documents", "metadatas"])
        if not result.get("ids"):
            return []
        return [
            {
                "id": result["ids"][i],
                "text": result["documents"][i],
                "metadata": result["metadatas"][i] or {},
            }
            for i in range(len(result["ids"]))
        ]

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Void / unvoid (soft delete with tombstone)
    # ------------------------------------------------------------------
    def void(
        self,
        memory_ids: List[str],
        void_reason: str,
        voided_by: str,
        void_group_id: Optional[str] = None,
    ) -> int:
        if not memory_ids:
            return 0
        result = self._collection.get(
            ids=memory_ids,
            include=["documents", "metadatas", "embeddings"],
        )
        if not result.get("ids"):
            return 0

        group_id = void_group_id or str(uuid.uuid4())
        voided_at = _now_iso()
        with open(self._tombstone_path, "a", encoding="utf-8") as f:
            for i, mid in enumerate(result["ids"]):
                entry = {
                    "memory_id": mid,
                    "text": result["documents"][i],
                    "metadata": result["metadatas"][i] or {},
                    "embedding": list(result["embeddings"][i]),
                    "voided_at": voided_at,
                    "voided_by": voided_by,
                    "void_reason": void_reason,
                    "void_group_id": group_id,
                }
                f.write(json.dumps(entry) + "\n")

        self._collection.delete(ids=result["ids"])
        return len(result["ids"])

    def void_by_source_message(
        self, source_message_id: str, voided_by: str
    ) -> Optional[str]:
        existing = self.get_all_by_source_message(source_message_id)
        if not existing:
            return None
        group_id = str(uuid.uuid4())
        self.void(
            memory_ids=[m["id"] for m in existing],
            void_reason=f"message {source_message_id}",
            voided_by=voided_by,
            void_group_id=group_id,
        )
        return group_id

    def void_by_author(
        self, author_id: str, voided_by: str
    ) -> tuple[Optional[str], int]:
        memories = self.get_by_author(author_id)
        if not memories:
            return None, 0
        group_id = str(uuid.uuid4())
        ids = [m["id"] for m in memories]
        count = self.void(
            memory_ids=ids,
            void_reason=f"author {author_id}",
            voided_by=voided_by,
            void_group_id=group_id,
        )
        return group_id, count

    def _read_tombstones(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._tombstone_path):
            return []
        entries = []
        with open(self._tombstone_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _write_tombstones(self, entries: List[Dict[str, Any]]) -> None:
        tmp = self._tombstone_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        os.replace(tmp, self._tombstone_path)

    def find_voided_by_source_message(
        self, source_message_id: str
    ) -> List[Dict[str, Any]]:
        return [
            e for e in self._read_tombstones()
            if e.get("metadata", {}).get("source_message_id") == str(source_message_id)
        ]

    def find_voided_by_author(self, author_id: str) -> List[Dict[str, Any]]:
        return [
            e for e in self._read_tombstones()
            if e.get("metadata", {}).get("author_id") == str(author_id)
        ]

    def unvoid_group(self, void_group_id: str) -> int:
        all_entries = self._read_tombstones()
        to_restore = [e for e in all_entries if e.get("void_group_id") == void_group_id]
        if not to_restore:
            return 0

        for entry in to_restore:
            meta = dict(entry.get("metadata", {}) or {})
            meta["unvoided_at"] = _now_iso()
            self._collection.add(
                ids=[entry["memory_id"]],
                embeddings=[entry["embedding"]],
                documents=[entry["text"]],
                metadatas=[meta],
            )

        remaining = [e for e in all_entries if e.get("void_group_id") != void_group_id]
        self._write_tombstones(remaining)
        return len(to_restore)

    def purge_expired_tombstones(self, retention_days: int) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        all_entries = self._read_tombstones()
        kept, expired = [], []
        for e in all_entries:
            try:
                voided_at = datetime.fromisoformat(e["voided_at"])
                if voided_at < cutoff:
                    expired.append(e)
                else:
                    kept.append(e)
            except (KeyError, ValueError):
                kept.append(e)
        if expired:
            self._write_tombstones(kept)
        return expired

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
