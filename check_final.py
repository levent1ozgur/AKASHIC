"""Debug: what exactly does the API return for FINALKEY_TWO after delete?"""
import json, urllib.request

API = "http://127.0.0.1:18765"
req = urllib.request.Request(f"{API}/query", data=json.dumps({"query":"FINALKEY_TWO","route":"vault","top_k":3}).encode(), headers={"Content-Type":"application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=30).read())

print(f"Chunks: {len(r.get('chunks',[]))} cache_hit={r.get('cache_hit',False)}")
for c in r.get("chunks",[]):
    print(f"  source={c['source']!r}")
    print(f"  text={c['text'][:100]!r}")
    print()
