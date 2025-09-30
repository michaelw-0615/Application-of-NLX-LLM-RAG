#!/usr/bin/env python
# coding: utf-8

# ## Step 5: Advanced RAG Pipeline Implementation
# 
# Features:
#  1) Multi-Query Expansion (MQE) with Flan-T5 → broaden recall
#  2) Cross-Encoder reranking (ms-marco MiniLM-L-6-v2) → improve precision
# 
#  Pipeline:
#   - Build FAISS index using all-MiniLM-L6-v2 embeddings
#   - For each question: generate rewrites → retrieve candidates
#   - Rerank with CrossEncoder → compose grounded context (citations)
#   - Generate final answer (Flan-T5 by default; OpenAI optional)
#   - (Optional) small SQuAD evaluation for sanity check

# ## 0) Configs

# In[1]:


CFG = {
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim embeddings
    "chunk_max_chars": 600,        # simple character-based chunking
    "batch_size": 64,
    "retrieve_k_per_query": 20,    # candidates per (query/rewrite)
    "n_rewrites": 3,               # number of MQE rewrites
    "final_top_k": 5,              # passages kept after CE reranking
    "max_ctx_chars": 2000,         # context budget for the prompt
    "use_openai": False,           # switch to True to use OpenAI (needs OPENAI_API_KEY)
}
print("Config:", CFG)


# ## 1) Install and import packages

# In[2]:


import sys, subprocess, pkgutil, os, time
def ensure(pkg, pip_name=None):
    pip_name = pip_name or pkg
    if pkgutil.find_loader(pkg) is None:
        print("Installing:", pip_name)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
    else:
        print("OK:", pkg)

ensure("datasets", "datasets")
ensure("sentence_transformers", "sentence-transformers")
ensure("transformers", "transformers")
ensure("faiss", "faiss-cpu")
ensure("numpy", "numpy")
ensure("pandas", "pandas")
ensure("evaluate", "evaluate")

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import faiss, numpy as np, pandas as pd
import evaluate


# ## 2) Dataset loading and chunking

# In[14]:


ds_corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
print("Corpus Dataset:", ds_corpus)
corpus = ds_corpus["passages"]

ds_qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
print("QA Dataset:", ds_qa)
qa = ds_qa["test"]


def simple_chunks(text, max_chars=CFG["chunk_max_chars"]):
    if not text:
        return []
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

docs = []
for i, row in enumerate(corpus):
    t = row.get("text") or row.get("passage") or ""
    for j, ch in enumerate(simple_chunks(t, CFG["chunk_max_chars"])):
        docs.append({"id": f"{i}-{j}", "text": ch})

print("Total chunks:", len(docs))
print("Example chunk:", docs[0]["id"], docs[0]["text"][:120], "...")


# ## 3) Embedding Index (FAISS, cosine via IP on normalized vecs)

# In[21]:


embed_model = SentenceTransformer(CFG["embed_model"])
texts = [d["text"] for d in docs]
Emb = embed_model.encode(
    texts, batch_size=CFG["batch_size"], show_progress_bar=True,
    normalize_embeddings=True, truncate=True
).astype("float32")

index = faiss.IndexFlatIP(Emb.shape[1])
index.add(Emb)
id_map = np.array([d["id"] for d in docs])
print("Index built | dim:", Emb.shape[1], "| num_vecs:", index.ntotal)


# ## 4) Advanced feature 1: Query Rewriting and Multi-Query Expansion (MQE)
# - Use Flan-T5 as lightweight transformer
# - Featuring diversified rewrite and deduplication

# In[37]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
import re


gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
generator = pipeline("text2text-generation", model=gen_mdl, tokenizer=gen_tok)

def _clean(q: str) -> str:
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    q = q.replace("’","'").replace("“","\"").replace("”","\"").replace("–","-").replace("—","-")
    return q

def _semantic_dedup(cands, embed_model, sim_thresh=0.95, keep_n=3):
    """Deduplication using sentence-transformer"""
    uniq = []
    if not cands:
        return uniq
    vecs = embed_model.encode(cands, normalize_embeddings=True).astype("float32")
    for i, c in enumerate(cands):
        if not uniq:
            uniq.append((c, vecs[i]));
            continue
        sims = [float(vecs[i] @ v) for _, v in uniq]
        if max(sims) < sim_thresh:
            uniq.append((c, vecs[i]))
        if len(uniq) >= keep_n:
            break
    return [t[0] for t in uniq]

def make_rewrites(question, n=CFG["n_rewrites"]):
    """Generate n diverse rewrites of the question"""
    q0 = _clean(question)
    prompt = (
        "Rewrite the question into diverse, retrieval-friendly queries. "
        "Return short alternatives, one per line, no numbering.\n\n"
        f"Question: {q0}"
    )

    # beams >= n to ensure num_return_sequences ≤ num_beams
    N = max(n, 6)
    try:
        outs = generator(
            prompt,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_beams=N,                # beams >= num_return_sequences
            num_return_sequences=N,     # Return N candidates at a time
            no_repeat_ngram_size=3,     # Limit repetition
        )
    except ValueError as e:

        N = max(n, 4)
        outs = generator(
            prompt,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_beams=N,
            num_return_sequences=N,
            no_repeat_ngram_size=3,
        )

    # Summarization
    raw_lines = []
    for o in outs:
        for ln in o["generated_text"].splitlines():
            ln = _clean(ln)
            if ln:
                raw_lines.append(ln)

    # Dedupe candidates identical to original questions
    uniq_lines, seen = [], set()
    for s in raw_lines:
        if s.lower() == q0.lower():
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        uniq_lines.append(s)

    # Semantic dedupe
    diverse = _semantic_dedup(uniq_lines, embed_model, sim_thresh=0.95, keep_n=max(n,1))

    while len(diverse) < n:
        diverse.append(q0)

    return diverse[:n]


# ## 5) Candidate retrieval across rewrites

# In[38]:


def faiss_search(query, top_k):
    qv = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, top_k)
    return list(zip(I[0].tolist(), D[0].tolist()))  # (idx, score)

def gather_candidates(question):
    """
    For original question + rewrites, retrieve candidates and deduplicate by passage index.
    Keep max similarity for each idx; return list sorted by vector sim desc.
    """
    rewrites = make_rewrites(question, CFG["n_rewrites"])
    cand = []
    for q in [question] + rewrites:
        hits = faiss_search(q, CFG["retrieve_k_per_query"])
        for idx, s in hits:
            cand.append((idx, s, q))
    best = {}
    for idx, s, q in cand:
        if idx not in best or s > best[idx][0]:
            best[idx] = (s, q)
    out = [(k, v[0], v[1]) for k, v in best.items()]
    out.sort(key=lambda x: x[1], reverse=True)
    return out, rewrites


# ## 6) Advanced feature 2: Cross-Encoder Reranking
# - A strong MS MARCO cross-encoder for (query, passage) scoring
# - Explicit truncation to 512 tokens

# In[42]:


from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
import numpy as np

# Cross encoder and the corresponding auto tokenizer
ce_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ce_tok = AutoTokenizer.from_pretrained(ce_model_id)

# Manually defined max length
reranker = CrossEncoder(ce_model_id, max_length=512)

def truncate_by_tokens(text: str, tokenizer, max_len: int) -> str:
    """Truncate text to max_len"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_len:
        ids = ids[:max_len]
    return tokenizer.decode(ids, skip_special_tokens=True)

def rerank(
    question: str,
    candidates,                   # list of (idx, vec_sim, rewrite_used)
    top_k: int = CFG["final_top_k"],
    q_budget: int = 32,           # Token budget for questions
    p_budget: int = 480           # Token budget for text passage
):
    """
    Token-based trunking for questions and text
    return：list[(idx, rerank_score)]
    """
    # 1) Truncate questions
    q_trunc = truncate_by_tokens(question, ce_tok, max_len=q_budget)

    # 2) Perform pairing and truncate passage to p-budget
    pairs, idxs = [], []
    for idx, _, _ in candidates:
        p_text = docs[idx]["text"]
        p_trunc = truncate_by_tokens(p_text, ce_tok, max_len=p_budget)
        pairs.append((q_trunc, p_trunc))
        idxs.append(idx)

    if not pairs:
        return []

    # 3) Cross-encoder ranking
    scores = reranker.predict(
        pairs,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    )

    # 4) Fetch top-k
    order = np.argsort(-scores)[:top_k]
    chosen = [(int(idxs[i]), float(scores[i])) for i in order]
    return chosen


# ## 7) Build context with citations & generate

# In[43]:


def build_context_with_citations(chosen, budget=CFG["max_ctx_chars"]):
    parts, cites, used = [], [], 0
    for idx, score in chosen:
        text = docs[idx]["text"]
        pid = id_map[idx]
        tag = f"[{pid} | {score:.3f}]"
        snippet = f"{tag}\n{text}"
        if used + len(snippet) > budget:
            parts.append(snippet[: max(0, budget - used)])
            cites.append({"id": pid, "score": float(score)})
            break
        parts.append(snippet)
        cites.append({"id": pid, "score": float(score)})
        used += len(snippet)
    return "\n\n".join(parts), cites

def persona_prompt(context, question):
    return (
        "You are a concise encyclopedia editor. Answer USING ONLY the context; "
        "if insufficient, reply 'I don't know.' Include short inline source tags if helpful.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def answer_advanced(question):
    # 1) MQE + vector recall
    candidates, rewrites = gather_candidates(question)
    # 2) Cross-encoder rerank
    chosen = rerank(question, candidates, top_k=CFG["final_top_k"])
    # 3) Grounded context with citations
    ctx, citations = build_context_with_citations(chosen, CFG["max_ctx_chars"])

    if CFG["use_openai"] and os.getenv("OPENAI_API_KEY"): #Optional: use ChatGPT 4o if valid API key available
        from openai import OpenAI
        client = OpenAI()
        prompt = persona_prompt(ctx, question)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=256
        )
        ans = resp.choices[0].message.content.strip()
    else:
        prompt = persona_prompt(ctx, question)
        ans = generator(prompt, max_new_tokens=256)[0]["generated_text"].strip()

    return ans, citations, rewrites


# ## 8) Demonstration of advanced features

# In[44]:


sample_q = qa[0]["question"]
print("Question:", sample_q)
ans, cites, rewrites = answer_advanced(sample_q)
print("\nRewrites:\n", "\n".join(rewrites))
print("\nAnswer:\n", ans)
print("\nCitations:", cites[:CFG["final_top_k"]])

