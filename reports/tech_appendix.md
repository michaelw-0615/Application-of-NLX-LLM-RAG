# Technical Appendix
## Appendix A: AI Usage Log
| Tool             | Purpose | Input | Output Usage | Verification |
|-------------------------|--------------------------------|-------------|-------------|------------|
| ChatGPT 5 | Help understand project scope and structure of deliverables | (Assignment description summarized into .docx file) Help me analyze the deliverables for each step and the structure of the final GitHub repo. | Used as reference for repo creation | Manual check on project description and discussion with peers |
| ChatGPT 5 | Help understand differences of different vector DBs and make choices  | Compare FAISS and Milvus and list their respective advantages. | Comparison analysis and DB selection | Manual search into web pages and course materials |
| ChatGPT 5 | Analyze text chunking strategy | (Distribution graph of passage token lengths) Which one should I choose, 300, 600 or 1200, as maximum chunking length? | Used as parameter in Step 3 evaluation | Code testing on different parameters |
| ChatGPT 5 | Analyze input dimension error | In step 3, why can't I feed in a 526-dim text vector into SentenceTransformer? | Refine embedding and evaluation strategy | Code testing under different embedding strategies and model input limits |
| ChatGPT 5 | Advanced feature result analysis | During rewriting, why does the RAG model outputs exactly the same question? | Refine parameters for advanced pipeline | Code testing under different sets of parameters |
| ChatGPT 5| Debugging | Why is the RAGAS model reporting the error: "OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"?| Check on API access | Discussion with peers|
