"""Add version fields to AKASHIC: chunker.py, embedder.py, config."""
# === CHUNKER: Add version constants and fields ===
import re

chunker_path = '/home/user/Desktop/AKASHIC-main/pipeline/ingestion/chunker.py'
embedder_path = '/home/user/Desktop/AKASHIC-main/pipeline/ingestion/embedder.py'
config_path = '/home/user/Desktop/AKASHIC-main/config.yaml'

with open(chunker_path) as f:
    chunker_text = f.read()

# 1. Add version constants after the DEFAULT_* constants
chunker_text = chunker_text.replace(
    "DEFAULT_CHARS_PER_TOKEN = 4   # rough approximation for token estimation",
    """DEFAULT_CHARS_PER_TOKEN = 4   # rough approximation for token estimation

# Pipeline version constants — bump when the chunking or context logic changes
CHUNKING_VERSION          = "hierarchical/v3"
CONTEXT_ENRICHMENT_VERSION = "v2"          # section_context + heading_path enrichment"""
)

# 2. Add version fields to Chunk dataclass (after section_context)
chunker_text = chunker_text.replace(
    "section_context: str = \"\"          # e.g. \"Aloe > Indications\" (for flat docs)",
    """section_context: str = \"\"          # e.g. \"Aloe > Indications\" (for flat docs)
    chunking_version: str           = CHUNKING_VERSION
    context_enrichment_version: str = CONTEXT_ENRICHMENT_VERSION"""
)

with open(chunker_path, 'w') as f:
    f.write(chunker_text)
print("1/3: Chunker patched — version constants + Chunk fields")

# === EMBEDDER: Add embedding_model to metadata ===
with open(embedder_path) as f:
    embedder_text = f.read()

# Add embedding_model version to _chunk_to_metadata
old_meta = '''        \"created_at\":     chunk.created_at,
    }'''
new_meta = '''        \"created_at\":              chunk.created_at,
        \"chunking_version\":           chunk.chunking_version,
        \"context_enrichment_version\": chunk.context_enrichment_version,
    }'''
embedder_text = embedder_text.replace(old_meta, new_meta)

# Also add the model name from the embedder's config
# Find the _embed_texts call or add to the embed method
# Actually, we need to pass the model name to _chunk_to_metadata
# Better approach: add a module-level function that reads from config or use a constant
# Simplest: add it in the metadata dict in the embed method's batch loop
old_metadatas = 'metadatas  = [_chunk_to_metadata(c) for c in batch]'
new_metadatas = '''metadatas  = [_chunk_to_metadata(c) for c in batch]
            # Tag each chunk with the embedding model that produced it
            for md in metadatas:
                md.update({"embedding_model": self.model})'''
embedder_text = embedder_text.replace(old_metadatas, new_metadatas)

with open(embedder_path, 'w') as f:
    f.write(embedder_text)
print("2/3: Embedder patched — version fields in metadata + embedding_model tag")

# === CONFIG: Add version tracking ===
with open(config_path) as f:
    config_text = f.read()

# Add version header
version_block = """
# --- Pipeline versions (bump when pipeline logic changes) ---
chunking_version:          hierarchical/v3
context_enrichment_version: v2
"""
if "# --- Pipeline versions" not in config_text:
    config_text += version_block
    print("3/3: Config updated with version info")
else:
    print("3/3: Config already has version info — skipping")

with open(config_path, 'w') as f:
    f.write(config_text)

print("\nDone. Verify: grep for CHUNKING_VERSION, embedding_model, chunking_version")
