"""Check what chunks are returned for DELKEY_X after delete."""
import json, urllib.request

API = "http://127.0.0.1:18765"
req = urllib.request.Request(f"{API}/query", data=json.dumps({"query":"DELKEY_X","route":"vault","top_k":3}).encode(), headers={"Content-Type":"application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=30).read())

print(f"Total chunks: {len(r.get('chunks',[]))}, cache_hit={r.get('cache_hit',False)}")
for c in r.get("chunks",[]):
    print(f"  source={c['source']}")
    print(f"  text={repr(c['text'][:80])}")
    print()
