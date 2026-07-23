"""Trace the lifecycle test step by step with registry checks."""
import os, time, json, urllib.request, re, sys, sqlite3
from pathlib import Path
sys.path.insert(0, ".")

API = "http://127.0.0.1:18765"
FILE = Path("/home/user/Documents/SecondBrain/._ftest.md")
FILE.unlink(missing_ok=True)

from storage.chroma import ChromaStore, COLLECTION_VAULT
chroma = ChromaStore("data/vector_db"); chroma.init()

def check_source(source):
    results = chroma._collection(COLLECTION_VAULT).get(include=["metadatas"])
    matches = [(cid, md) for cid, md in zip(results["ids"], results["metadatas"]) if md.get("source","") == source]
    return len(matches), [m[1].get("doc_id","?")[:12] for m in matches]

def check_reg(path):
    conn = sqlite3.connect("data/registry.db")
    cur = conn.execute("SELECT doc_id, source_path, status FROM documents WHERE source_path=?", (path,))
    row = cur.fetchone()
    conn.close()
    return row

def q(query):
    req = urllib.request.Request(f"{API}/query", data=json.dumps({"query":query,"route":"vault","top_k":3}).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

# CREATE
FILE.write_text("# Final Test\nFTKEY_A.\n")
time.sleep(10)
print("=== CREATE ===")
print(f"  Reg: {check_reg('._ftest.md')}")
n, ids = check_source("._ftest.md")
print(f"  ChromaDB: {n} chunks, doc_ids={ids}")

# MODIFY
FILE.write_text("# Final Test\nFTKEY_B.\n")
time.sleep(10)
print("\n=== MODIFY ===")
print(f"  Reg: {check_reg('._ftest.md')}")
n, ids = check_source("._ftest.md")
print(f"  ChromaDB: {n} chunks, doc_ids={ids}")
r_old = q("FTKEY_A")
r_new = q("FTKEY_B")
old_txts = [(c['source'], c['text'][:40]) for c in r_old.get('chunks',[])]
new_txts = [(c['source'], c['text'][:40]) for c in r_new.get('chunks',[])]
print(f"  Query FTKEY_A: {len(r_old['chunks'])} chunks, cache={r_old.get('cache_hit')}")
for s, t in old_txts: print(f"    [{s}] {t!r}")
print(f"  Query FTKEY_B: {len(r_new['chunks'])} chunks, cache={r_new.get('cache_hit')}")
for s, t in new_txts: print(f"    [{s}] {t!r}")

# RENAME
FILE2 = Path("/home/user/Documents/SecondBrain/._ftest_renamed.md")
FILE2.unlink(missing_ok=True)
os.rename(str(FILE), str(FILE2))
time.sleep(10)
print("\n=== RENAME ===")
print(f"  Reg old: {check_reg('._ftest.md')}")
print(f"  Reg new: {check_reg('._ftest_renamed.md')}")
n, ids = check_source("._ftest.md")
print(f"  ChromaDB OLD source: {n} chunks, doc_ids={ids}")
n, ids = check_source("._ftest_renamed.md")
print(f"  ChromaDB NEW source: {n} chunks, doc_ids={ids}")
r = q("FTKEY_B")
all_srcs = [(c['source'], c['text'][:40]) for c in r.get('chunks',[])]
print(f"  Query FTKEY_B: {len(r['chunks'])} chunks, cache={r.get('cache_hit')}")
for s, t in all_srcs: print(f"    [{s}] {t!r}")

# DELETE
FILE2.unlink()
time.sleep(10)
print("\n=== DELETE ===")
print(f"  Reg: {check_reg('._ftest_renamed.md')}")
n, ids = check_source("._ftest_renamed.md")
print(f"  ChromaDB: {n} chunks")
n, ids = check_source("._ftest.md")
print(f"  ChromaDB (old source): {n} chunks")
r = q("FTKEY_B")
found_exact = lambda tok, resp: bool(re.compile(r"\b" + re.escape(tok) + r"\b").search(str(resp.get("chunks",[]))))
fe = found_exact("FTKEY_B", r)
print(f"  Query FTKEY_B: {len(r['chunks'])} chunks, cache={r.get('cache_hit')}, found_exact={fe}")
for c in r.get('chunks',[]):
    print(f"    [{c['source']}] {c['text'][:60]!r}\n      doc_id={c.get('doc_id','?')[:12]}...")

FILE.unlink(missing_ok=True); FILE2.unlink(missing_ok=True)
print("\nDone")
