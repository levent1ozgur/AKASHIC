"""Direct delete test: create, delete, check registry and ChromaDB step by step."""
import os, time, json, urllib.request, re, sys, sqlite3
from pathlib import Path
sys.path.insert(0, ".")

API = "http://127.0.0.1:18765"
VAULT = Path("/home/user/Documents/SecondBrain")
FILE = VAULT / ".deltest.md"
FILE.unlink(missing_ok=True)

from storage.chroma import ChromaStore, COLLECTION_VAULT
from storage.bm25 import BM25Store, INDEX_VAULT
chroma = ChromaStore("data/vector_db"); chroma.init()
bm25 = BM25Store("data/bm25"); bm25.init()

def q(query):
    req = urllib.request.Request(f"{API}/query", data=json.dumps({"query":query,"route":"vault","top_k":3}).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def count_chunks(source):
    results = chroma._collection(COLLECTION_VAULT).get(include=["metadatas"])
    pat = source
    return sum(1 for md in results["metadatas"] if md.get("source","") == pat)

# CREATE
FILE.write_text("# Del Test\nDELKEY_X.\n")
time.sleep(10)

conn = sqlite3.connect("data/registry.db")
rows = conn.execute("SELECT source_path,status FROM documents WHERE source_path LIKE '%.deltest%'").fetchall()
conn.close()
print("CREATE: registry:", rows)

chunk_count = count_chunks(".deltest.md")
print(f"CREATE: ChromaDB entries for .deltest.md: {chunk_count}")

r = q("DELKEY_X")
print(f"CREATE: query: {len(r.get('chunks',[]))} chunks, cache_hit={r.get('cache_hit',False)}")

# DELETE
FILE.unlink()
time.sleep(10)

conn = sqlite3.connect("data/registry.db")
rows = conn.execute("SELECT source_path,status FROM documents WHERE source_path LIKE '%.deltest%'").fetchall()
conn.close()
print("\nDELETE: registry:", rows)

chunk_count = count_chunks(".deltest.md")
print(f"DELETE: ChromaDB entries for .deltest.md: {chunk_count}")

r = q("DELKEY_X")
print(f"DELETE: query: {len(r.get('chunks',[]))} chunks, cache_hit={r.get('cache_hit',False)}")

FILE.unlink(missing_ok=True)
