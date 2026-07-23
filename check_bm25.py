"""Diagnose BM25 vault index state."""
import json, sys
sys.path.insert(0, '.')

from storage.bm25 import BM25Store, INDEX_VAULT

bm25 = BM25Store("data/bm25")
bm25.init()

state = bm25._check(INDEX_VAULT)
print(f"BM25 vault index:")
print(f"  chroma_ids count: {len(state.chroma_ids)}")
print(f"  texts count: {len(state.texts)}")
print(f"  metadatas count: {len(state.metadatas)}")

for i in range(min(5, state.size)):
    cid = state.chroma_ids[i][:16]
    txt = state.texts[i][:40]
    src = state.metadatas[i].get("source", "?")
    print(f"  [{i}] id={cid}... text={txt}... source={src}")
