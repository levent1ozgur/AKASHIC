"""Clean ALL lifecycle test artifacts from ChromaDB, BM25, and registry."""
import sys, os
sys.path.insert(0, ".")

from storage.chroma import ChromaStore, COLLECTION_VAULT
from storage.bm25 import BM25Store, INDEX_VAULT
from storage.registry import Registry

chroma = ChromaStore("data/vector_db"); chroma.init()
bm25 = BM25Store("data/bm25"); bm25.init()
reg = Registry("data/registry.db"); reg.init()

# All lifecycle test source patterns
results = chroma._collection(COLLECTION_VAULT).get(include=["metadatas"])
removed = set()

for cid, md in zip(results["ids"], results["metadatas"]):
    src = md.get("source", "")
    # Match any lifecycle/test source
    if any(p in src for p in [".debug", ".lctest", ".deltest", ".ftest", "lifecycle_test", "_lctest", "_lifecycle_"]):
        doc_id = md.get("doc_id", "")
        chroma.delete_by_metadata(COLLECTION_VAULT, {"doc_id": doc_id})
        bm25.delete_by_doc_id(INDEX_VAULT, doc_id)
        reg.delete_document(doc_id)
        removed.add(src)
        print(f"Removed: {src}")

# Also clean registry entries by source path pattern
conn = __import__("sqlite3").connect("data/registry.db")
cur = conn.execute("SELECT doc_id, source_path FROM documents WHERE source_path LIKE '%.debug%' OR source_path LIKE '%.lctest%' OR source_path LIKE '%.deltest%' OR source_path LIKE '%.ftest%'")
for row in cur.fetchall():
    doc_id, src = row
    chroma.delete_by_metadata(COLLECTION_VAULT, {"doc_id": doc_id})
    bm25.delete_by_doc_id(INDEX_VAULT, doc_id)
    reg.delete_document(doc_id)
    if src not in removed:
        print(f"Also cleaned: {src} (from registry)")
conn.close()

# Cleanup actual files from vault
base = os.path.expanduser("~/Documents/SecondBrain")
for f in os.listdir(base):
    if f.startswith(".debug") or f.startswith(".lctest") or f.startswith(".deltest") or f.startswith(".ftest") or f.startswith("_lctest") or f.startswith("_lifecycle_"):
        fp = os.path.join(base, f)
        os.unlink(fp)
        print(f"Deleted file: {f}")

print(f"\nVault collection now: {chroma.count(COLLECTION_VAULT)} entries")
