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

## 
