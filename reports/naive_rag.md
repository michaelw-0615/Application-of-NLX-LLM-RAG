# Step 2: Naive RAG Implementation and Code Documentation

## Overview
The purpose of this step is to design and implement a naive RAG pipeline that serves as the basis for later stages of advanced RAG features. Such a pipeline should be able to perform text data reading from the source dataset, text-based chunking, text embedding, data insertion, OpenAI API call, query and answer generation. 

## Dependencies and Environment
### Packages
- dataset
- sentence-transformers/all-MiniLM-L6-v2
- pymilvus (Milvus Lite)
- numpy
- pandas
- openai
### Environment Setup
- Python>=3.9
- Random seed set to 42 to ensure reproducibility
- Milvus connection is achieved via `uri="milvus.db"`
- Jupyter notebook supports execution in local environment, but we conducted evaluation and experiments in Google Colab

## Key Pipeline Configurations
- `chunk_max_chars`: the maximum length for text-based chunking, default set to 300. This is based on statistical results on `text-corpus` samples in Step 1.
- `model`: embedding model, set as `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"`; dimension set to 384.
- `uri`: Milvus Lite URI, set to `"milvus.db"`.
- `top_k`: top-K value, default set to 5. Will be reset to 1 during Step 3 initial evaluation.
- `temperature`: temperature for answer generation, default set to 0.2.
- `max_ctx_chars`: Maximum answer length, default set to 2000 characters.

## Data Loading
Per requirements, the `rag-datasets/rag-mini-wikipedia` dataset is used. Both subsets--`text-corpus` and `question-answer` are loaded, with the former used for embedding, database establishment and query, and the latter used for questioning during evaluation. 

## Text-Based Chunking
The method `simple_chunks(text, max_chars)` performs chunking on `text-corpus` text segments with a default maximum length of 300. It returns a list of dictionaries `[{id, text}]` which aligns with the distribution of tokens discovered in Step 1. In total, 3,299 id-text pairs will be generated and stored in the local database.

## Embedding and Vector Storage
`SentenceTransformer(all-MiniLM-L6-v2)` is used to embed chunked text segments with a batch size of 64. We also use L2 normalization to process embedding vectors so that cosine distance could be applied later. Milvus Lite is used to store data as `{id, embeddings, text}` dictionaries, with FLAT indexing to ensure precise query results.

## Retrieval, Prompting and Generation
The method `retrieve(query, top_k, ef)` is used to search for top-K answers in the format of `[(id, text, score)]` where `score` is derived from cosine similarity. The method `answer_with_context(query, top_k, max_ctx_chars)` uses OpenAI API (currently hard-coded) to generate top-k answers with maximum length `max_ctx_chars`. 
