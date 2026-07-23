"""Final lifecycle debug — long waits, cold cache, direct ChromaDB checks."""
import os, time, json, urllib.request, re, sys, subprocess
from pathlib import Path
sys.path.insert(0, ".")

API = "http://127.0.0.1:18765"
FILE = Path("/home/user/Documents/SecondBrain/._final_debug.md")
FILE.unlink(missing_ok=True)

# Kill API and restart (clear cache)
subprocess.run(["fuser", "-k", "18765/tcp"], capture_output=True)
time.sleep(2)
proc = subprocess.Popen(
    ["/home/user/Desktop/AKASHIC-main/venv/bin/python3", "-m", "uvicorn", "api.main:app",
     "--host", "127.0.0.1", "--port", "18765", "--log-level", "error"],
    cwd="/home/user/Desktop/AKASHIC-main",
    env={**os.environ, "PYTHONPATH": "/home/user/Desktop/AKASHIC-main"},
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
print(f"API started: PID {proc.pid}")
time.sleep(10)

# Verify API is up
try:
    urllib.request.urlopen(f"{API}/status", timeout=5)
    print("API ready")
except:
    print("API failed to start")
    sys.exit(1)

from storage.chroma import ChromaStore, COLLECTION_VAULT
from storage.bm25 import BM25Store, INDEX_VAULT

chroma = ChromaStore("data/vector_db"); chroma.init()
bm25 = BM25Store("data/bm25"); bm25.init()

def q(query):
    req = urllib.request.Request(f"{API}/query", data=json.dumps({"query":query,"route":"vault","top_k":3}).encode(), headers={"Content-Type":"application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    print(f"  cache_hit={d.get('cache_hit',False)}")
    return d

def found_exact(token, resp):
    pat = re.compile(r"\b" + re.escape(token) + r"\b")
    for c in resp.get("chunks", []):
        if pat.search(c.get("text", "")):
            return True, c["text"][:60]
    return False, ""

def check_index(source_path):
    """Check what ChromaDB has for this source."""
    results = chroma._collection(COLLECTION_VAULT).get(include=["documents", "metadatas"])
    for cid, doc, md in zip(results["ids"], results["documents"], results["metadatas"]):
        if md.get("source","") == source_path:
            print(f"  ChromaDB: text={repr(doc[:60])}")
            return True
    print(f"  ChromaDB: no entry")
    return False

# === 1. CREATE ===
print("\n=== CREATE ===")
FILE.write_text("# Final Debug\nFINAL_ONE.\n")
time.sleep(10)

check_index("._final_debug.md")
r = q("FINAL_ONE")
hit, txt = found_exact("FINAL_ONE", r)
print(f"  FOUND Final_ONE in API: {hit}")

# === 2. MODIFY ===
print("\n=== MODIFY ===")
FILE.write_text("# Final Debug\nFINAL_TWO.\n")
time.sleep(10)

check_index("._final_debug.md")
r1 = q("FINAL_ONE")
r2 = q("FINAL_TWO")
hit1, txt1 = found_exact("FINAL_ONE", r1)
hit2, txt2 = found_exact("FINAL_TWO", r2)
print(f"  FOUND Final_ONE: {hit1} (should be False)")
print(f"  FOUND Final_TWO: {hit2} (should be True)")
if hit1:
    print(f"  FALSE MATCH text: {txt1}")

# === 3. DELETE ===
print("\n=== DELETE ===")
FILE.unlink()
time.sleep(10)

check_index("._final_debug.md")
r = q("FINAL_TWO")
hit, txt = found_exact("FINAL_TWO", r)
print(f"  FOUND Final_TWO after delete: {hit} (should be False)")
if hit:
    print(f"  FALSE MATCH text: {txt}")

# Cleanup
FILE.unlink(missing_ok=True)
print("\nDone")
