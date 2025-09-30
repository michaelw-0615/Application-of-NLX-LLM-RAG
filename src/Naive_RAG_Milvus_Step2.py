#!/usr/bin/env python
# coding: utf-8

# 
# # Step 2 — Naive RAG (Milvus Lite Edition)
# 
# This notebook is **self-contained** and can be run independently of any Step 1 notebook.  
# It will:
# 
# 1. Load the **RAG Mini Wikipedia** dataset from Hugging Face
# 2. (Optionally) chunk passages to manageable sizes
# 3. Generate embeddings with **sentence-transformers/all-MiniLM-L6-v2** (384-dim)
# 4. Create a **Milvus Lite** collection, index vectors, and insert data
# 5. Implement a simple **retrieve** function (ANN search with HNSW + IP on normalized vectors)
# 6. Implement a minimal **answer_with_context** function that retrieves top-k passages and calls an LLM
# 7. Run an end-to-end demo using a question from the QA split
# 
# > **Note**: If you prefer FAISS, you can swap the vector store step for a FAISS index.  
# > **Milvus Lite** runs locally via a single file (e.g., `milvus.db`) using `pymilvus`.
# 

# 
# ## 0. (Optional) Install Dependencies
# 
# Run this cell if your environment doesn't already have these packages.  
# Restart kernel if upgrades are performed.
# 

# In[ ]:


import sys, subprocess, pkgutil

def pip_install(pkg):
    print(f"Installing: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

need = {
    "datasets": "datasets",
    "sentence_transformers": "sentence-transformers",
    "pymilvus": "pymilvus>=2.4.0",
    "numpy": "numpy",
    "pandas": "pandas",
    "openai": "openai",  # only if you plan to use OpenAI
}

for mod, pip_name in need.items():
    if pkgutil.find_loader(mod) is None:
        pip_install(pip_name)
    else:
        print(f"OK: {mod}")


# 
# ## 1. Imports & Basic Setup
# 

# In[ ]:


from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Reproducibility
import random
random.seed(42)
np.random.seed(42)


# 
# ## 2. Load the RAG Mini Wikipedia Dataset
# 
# We load two splits:
# - `text-corpus`: documents/passages to be indexed
# - `question-answer`: evaluation questions (we'll sample one to demo)
# 

# In[ ]:


ds_corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
# print(ds_corpus.keys()) # Already diagnosed, no need to print again
corpus = ds_corpus["passages"] # Corrected key
ds_qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
# print(ds_qa.keys()) # Check keys for qa dataset as well
qa = ds_qa["test"] # Corrected key


print(corpus)
print(qa)

# Extract raw texts
raw_passages = [row.get("text") or row.get("passage") or "" for row in corpus]
print("Sample passage:", raw_passages[0][:300], "...")
print("Total passages:", len(raw_passages))


# 
# ## 3. Character-Based Chunking
# 
# 
# 

# In[ ]:


def simple_chunks(text, max_chars=600):
    # Avoid empty chunks
    if not text:
        return []
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

docs = []
for i, t in enumerate(raw_passages):
    chunks = simple_chunks(t)
    for j, ch in enumerate(chunks):
        docs.append({"id": f"{i}-{j}", "text": ch})

print("Total chunks:", len(docs))
print("Sample chunk:", docs[0]["text"][:200], "...")


# 
# ## 4. Embeddings with all-MiniLM-L6-v2
# 
# Per assignment requirements, we use **sentence-transformers/all-MiniLM-L6-v2** (384-dim).  
# We also **L2-normalize** embeddings so that **Inner Product (IP)** becomes cosine similarity.
# 

# In[ ]:


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text_list = [d["text"] for d in docs]

# Encode in batches to reduce memory usage
emb = model.encode(text_list, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
emb = emb.astype("float32")  # Milvus expects float vectors

dim = emb.shape[1]
assert dim == 384, f"Unexpected dim {dim}" #
print("Embeddings shape:", emb.shape)


# 
# ## 5. Milvus Lite: Connect, Define Schema, Create Collection & Index
# 
# - Connect to Milvus Lite (embedded mode) via `uri="milvus.db"`
# - Define a schema with:
#   - `id` (primary key, VARCHAR)
#   - `embedding` (FLOAT_VECTOR, dim=384)
#   - `text` (VARCHAR, carry the chunk text)
# - Create an HNSW index with IP metric (on normalized vectors)
# 

# In[ ]:


# Connect (Milvus Lite will create a local file if it doesn't exist)
connections.connect(alias="default", uri="milvus.db")
print("Connected to Milvus Lite:", connections.has_connection("default"))

COLL_NAME = "rag_mini_wiki_chunks"

# Drop if exists (to make the notebook idempotent)
if utility.has_collection(COLL_NAME):
    utility.drop_collection(COLL_NAME)

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),  # fits our 1200-char chunks
]
schema = CollectionSchema(fields, description="Naive RAG chunks")
col = Collection(COLL_NAME, schema=schema, consistency_level="Strong")
print("Created collection:", col.name)

# Create index for ANN search
index_params = {
    "index_type": "FLAT", # Changed from HNSW to FLAT
    "metric_type": "IP",
    # Removed HNSW specific params
}
col.create_index(field_name="embedding", index_params=index_params)
print("Index created.")

# Load the collection to memory to serve queries
col.load()


# In[ ]:


import sys, subprocess
import pymilvus # Add this import
print("Installing pymilvus[milvus_lite]...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pymilvus[milvus_lite]"])
print("Installation complete.")


# 
# ## 6. Insert Data into Milvus
# 
# We insert `id`, `embedding`, and `text` columns in aligned order.
# 

# In[ ]:


# Prepare entities
ids = [d["id"] for d in docs]
texts = [d["text"] for d in docs]
entities = [ids, emb.tolist(), texts]

# Insert and flush
mr = col.insert(entities)
col.flush()

print("Inserted rows:", mr.insert_count)
print("Collection num_entities:", col.num_entities)


# 
# ## 7. Define a Retrieve Function
# 
# - Encode the query with the same embedding model
# - Search top-k neighbors using `col.search` on the vector field
# - Use IP metric (cosine similarity on normalized vectors)
# 

# In[ ]:


def retrieve(query, top_k=5, ef=64, output_fields=("id","text")):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")[0].tolist()
    search_params = {"metric_type": "IP", "params": {"ef": ef}}
    res = col.search(
        data=[qv],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=list(output_fields),
    )
    hits = []
    for h in res[0]:
        eid = h.entity.get("id")
        etext = h.entity.get("text")
        hits.append((eid, etext, float(h.distance)))
    return hits

# Quick smoke test
print(retrieve("What is the capital of France?", top_k=3))


# 
# ## 8. Minimal Generation with Context
# 
# This cell provides two options:
# - **OpenAI**: If `OPENAI_API_KEY` is set, call a chat completion model (e.g., `gpt-4o-mini`).  
# - **Local fallback**: If no key is present, return a template answer with the top context (for testing the pipeline).
# 

# In[ ]:


def answer_with_context(query, top_k=5, max_ctx_chars=2000):
    hits = retrieve(query, top_k=top_k)
    context = "\n\n".join([h[1] for h in hits])[:max_ctx_chars]

    try:
        from openai import OpenAI
        #api_key = os.getenv("OPENAI_API_KEY")
        api_key = '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'
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY in env")
        client = OpenAI(api_key=api_key)

        prompt = (
            "You are a helpful assistant. Answer strictly using the provided context. "
            "If the context is insufficient, answer 'I don't know.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        # Local fallback when no API key is present
        answer = (
            "[Local fallback] No OpenAI key detected. Sample answer based on top context snippet:\n\n"
            + context[:500] + ("..." if len(context) > 500 else "")
        )
    return answer, hits


# 
# ## 9. End-to-End Test
# 
# In order to test the effectiveness and robustness of the pipeline, we pick a sample question from the QA split and run the full pipeline.
# 

# In[ ]:


sample_q = qa[0]["question"]
print("Question:", sample_q)

answer, hits = answer_with_context(sample_q, top_k=5)
print("\n=== Answer ===\n", answer)
print("\n=== Top Hits (id, score) ===")
for eid, etxt, score in hits:
    print(eid, "| score:", round(score, 4), "| text:", (etxt[:120] + "..."))

