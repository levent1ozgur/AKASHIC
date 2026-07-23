"""Add query cache invalidation on vault document changes.
Patches api/main.py to clear the in-memory query cache when vault
documents are created, modified, renamed, or deleted.
"""
import sys, re

path = "/home/user/Desktop/AKASHIC-main/api/main.py"
with open(path) as f:
    content = f.read()

# 1. Add invalidate_query_cache function after state initialization
old = "    state.query_cache    = {}"
new = """    state.query_cache    = {}

def invalidate_query_cache() -> None:
    \"\"\"Clear the in-memory query cache when vault data changes.\"\"\"
    if state.query_cache:
        n = len(state.query_cache)
        state.query_cache.clear()
        logger.info("Query cache cleared (%d entries)", n)
"""

if old in content:
    content = content.replace(old, new, 1)
    print(f"  Added invalidate_query_cache()")
else:
    print(f"  WARNING: Could not find 'state.query_cache = {{}}'")

# 2. Add cache invalidation in vault event handlers
# In the DELETE handler, after removing the doc
old = """        if doc:
            state.chroma.delete_by_metadata(COLLECTION_VAULT, {\"doc_id\": doc.doc_id})
            state.bm25.delete_by_doc_id(INDEX_VAULT, doc.doc_id)
            state.registry.delete_document(doc.doc_id)
            state.graph.remove_node(rel_path)"""
new = """        if doc:
            state.chroma.delete_by_metadata(COLLECTION_VAULT, {\"doc_id\": doc.doc_id})
            state.bm25.delete_by_doc_id(INDEX_VAULT, doc.doc_id)
            state.registry.delete_document(doc.doc_id)
            state.graph.remove_node(rel_path)
            invalidate_query_cache()"""

if old in content:
    content = content.replace(old, new, 1)
    print(f"  Added cache invalidation to DELETE handler")
else:
    print(f"  WARNING: Could not find DELETE handler anchor")

# 3. In the MOVED handler, after removing old doc
old = """    if event.event_type == VaultEventType.MOVED:
        if event.old_path:
            old_doc = state.registry.get_document_by_path(event.old_path)
            if old_doc:
                state.registry.delete_document(old_doc.doc_id)
                state.graph.remove_node(event.old_path)"""
new = """    if event.event_type == VaultEventType.MOVED:
        if event.old_path:
            old_doc = state.registry.get_document_by_path(event.old_path)
            if old_doc:
                state.registry.delete_document(old_doc.doc_id)
                state.graph.remove_node(event.old_path)
                invalidate_query_cache()"""

if old in content:
    content = content.replace(old, new, 1)
    print(f"  Added cache invalidation to MOVED handler")
else:
    print(f"  WARNING: Could not find MOVED handler anchor")

# 4. In the MODIFY (and CREATE/MOVED-new) handler, after successful indexing
old = """        if chunks:
            state.embedder.embed(chunks, doc_id=doc_id)
            state.registry.update_status(doc_id, \"indexed\", chunk_count=len(chunks))
        else:
            state.registry.update_status(doc_id, \"failed\",
                                          error_message=\"No chunks produced.\")"""
new = """        if chunks:
            state.embedder.embed(chunks, doc_id=doc_id)
            state.registry.update_status(doc_id, \"indexed\", chunk_count=len(chunks))
            invalidate_query_cache()
        else:
            state.registry.update_status(doc_id, \"failed\",
                                          error_message=\"No chunks produced.\")"""

if old in content:
    content = content.replace(old, new, 1)
    print(f"  Added cache invalidation to CREATE/MODIFY handler")
else:
    print(f"  WARNING: Could not find CREATE/MODIFY handler anchor")

with open(path, "w") as f:
    f.write(content)

print(f"\nDone. Verifying...")
# Quick syntax check
import py_compile
try:
    py_compile.compile(path, doraise=True)
    print("  Syntax: OK")
except py_compile.PyCompileError as e:
    print(f"  Syntax ERROR: {e}")
