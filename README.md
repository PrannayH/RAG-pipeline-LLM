# RAGpipeline_LLM

LLMs have great potential in translation, essay writing, and general question-answering. But when it comes to domain-specific question-answering, they suffer from hallucinations. Besides, in a domain-specific QA app, only a few documents contain relevant context per query. So, we need a unified system that streamlines document extraction to answer generation and all the processes between them. This process is called Retrieval Augmented Generation.

Why RAG??
- Training Challenges:
    - LLMs like GPT-4 are trained on massive datasets, costing hundreds of millions of dollars.
    - Re-training such models on new data is impractical.
- Fine-tuning Complexity:
    - Fine-tuning is potent but resource-intensive in time and money.
    - Not practical unless there's a specific need.
- Prompting as an Alternative:
    - Prompting integrates new information into the LLM context window.
    - Effective for many real-life use cases, including document Q&A.
- RAG Efficiency:
    - Retrieval Augmented Generation (RAG) offers an efficient alternative for domain-specific Q&A.
    - Streamlines processes from document extraction to answer generation.
    - Particularly useful when only a few documents contain relevant context per query.

What Are The RAG Components?
In a typical RAG process, we have a few components.
• Text Splitter: Splits documents to accommodate context windows of LLMs.
• Embedding Model: The deep learning model used to get embeddings of documents.
• Vector Stores: The databases where document embeddings are stored and queried along with their metadata.
• LLM: The Large Language Model responsible for generating answers from queries.
• Utility Functions: This involves additional utility functions such as Webretriver and document parsers that aid in retrieving and pre-processing files.

In a typical RAG process. We have documents (PDFs, Web Pages, Docs), a tool to split large text into smaller chunks, embedding models to get vector representation of text chunks, Vector stores as knowledge bases, and an LLM to get answers from text chunks.

In our code:

1. Document Loading (`reader.py`):
   - Reads documents of various types using specific loaders.
   - Represents documents as instances of a custom `Document` class.

2. Text Chunking (`adapter.py`):
   - Breaks down loaded documents into smaller chunks.
   - Uses a text splitter with specified parameters.

3. Text Embedding (`embedder.py`):
   - Embeds text chunks using Hugging Face Transformers.
   - Supports embedding both individual chunks and query strings.

4. Vector Database (`vector_database.py`):
   - Manages a vector database using FAISS.
   - Inserts embeddings with metadata and performs similarity searches.

5. Language Model (`llm.py`):
   - Represents a language model (LLM) using GPT4All.
   - Instantiated with a model name and task, with a prompt template for response generation.

6. Pipeline Execution (`test.py`):
   - Orchestrates the entire RAG pipeline.
   - Loads, chunks, embeds, inserts into the vector database, queries the LLM, and searches for relevant chunks.
   - Exception handling for graceful error handling.

7. Common Themes:
   - Modular design with separate classes for each component.
   - Utilizes external libraries for embedding (Hugging Face Transformers) and vector database operations (FAISS).
   - Emphasis on exception handling for robustness.
