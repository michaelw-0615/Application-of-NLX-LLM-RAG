# Step 1: RAG Dataset Setup and Exploration

The RAG dataset we use for this project can be accessed via https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia. It is a text dataset splitted into two subsets: a) text-corpus, which contains 3,200 sentences or short terms excerpted from Wikipedia that describes facts or subjects; and b) question-answer, which contains 918 pairs of questions and ground-truth answers that are derived from the contents of text-corpus. Each subset has its own set of identifiers.

The passage in the text-corpus subset has no null or duplicate values. The lengths of text passages range from 1 to 2515, with an average length of 389.85 and a median of 299. Most entries fall into a moderate range of characters, though some are longer, which will influence our chunking strategy in later phases.
