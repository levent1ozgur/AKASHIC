"""Clean up old lifecycle test files from all AKASHIC indexes."""
import sys
sys.path.insert(0, ".")

from storage.chroma import ChromaStore, COLLECTION_VAULT
from storage.bm25 import BM25Store, INDEX_VAULT
from storage.registry import Registry

chroma = ChromaStore("data/vector_db"); chroma.init()
bm25 = BM25Store("data/bm25"); bm25.init()
reg = Registry("data/registry.db"); reg.init()

results = chroma._collection(COLLECTION_VAULT).get(include=["metadatas"])
test_sources = {
    "Daily/__lifecycle_test__.md",
    "_lctest2.md",
    "_lifecycle_test.md",
    "_lctest5.md",
    ".lctest_final.md",
    "_lctest3.md",
    "_lctest6.md",
    "._final_debug.md",
    "._lctest_deep.md",
    ".lctest_final.md",
}

removed = 0
for cid, md in zip(results["ids"], results["metadatas"]):
    if md.get("source", "") in test_sources:
        doc_id = md.get("doc_id", "")
        chroma.delete_by_metadata(COLLECTION_VAULT, {"doc_id": doc_id})
        bm25.delete_by_doc_id(INDEX_VAULT, doc_id)
        reg.delete_document(doc_id)
        removed += 1
        print(f"Removed {md['source']}: {doc_id[:12]}...")

print(f"\nRemoved {removed} document(s)")
print(f"Vault collection: {chroma.count(COLLECTION_VAULT)} entries")
print(f"Registry vault docs: {len(reg.get_all_vault_documents() if hasattr(reg,'get_all_vault_documents') else '?')}")
