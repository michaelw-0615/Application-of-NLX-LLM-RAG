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

## Configurations
