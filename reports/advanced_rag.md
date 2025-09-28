# Step 5: Advanced RAG Pipeline With Query Rewriting and Reranking

In this step, we build on the naive RAG framework implemented in Step 3 and add two advanced RAG features: query rewriting and reranking.

## Feature Selection Rationale
Multi-Query Expansion (MQE). In Step 4, experiments revealed a recall–precision trade-off: increasing top-k improved coverage but introduced noise, while smaller top-k missed relevant evidence. MQE mitigates this by generating multiple paraphrases of the input question. Each paraphrase is used for retrieval, broadening coverage without permanently bloating the context window. This reduces the risk of missing critical evidence due to lexical mismatch.
Cross-Encoder Reranking. Step 4 also showed that simple concatenation of top passages outperformed diversification methods like MMR, highlighting the importance of strict relevance. To further improve precision, a cross-encoder model (trained on MS MARCO relevance judgments) is introduced. Unlike bi-encoder embeddings, cross-encoders jointly encode the query and passage, producing a more fine-grained relevance score. This ensures that only the most relevant passages are selected for final generation, limiting context dilution.
