# Application-of-NLX-LLM-RAG
This is a project repository for *Assignment 2: Ground the Domain - From Naive RAG to Production Patterns* of CMU course 95702. This document serves as an introduction to the repo and an instruction for reproduction.

## Configuration and Requirements
- `python>= 3.11` for all scripts
- To reproduce each step using the Jupyter notebooks, simply download the respective files from `./notebooks`, upload to Google Colab or a local IDE to start running.
- To run each step as Python files, download the respective files from  `./src` and execute the following commands in PowerShell:
```powershell
pip install datasets evaluate numpy pandas matplotlib ragas "langchain-openai>=0.1.7" transformers sentence-transformers pymilvus faiss-cpu
python <file_path>/<file_name>.py


