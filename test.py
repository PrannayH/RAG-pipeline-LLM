# import os
# import sys
# from reader.reader import reader
# from adapter.adapter import adapter
# from embedder.embedder import embedder
# from vector_database.vector_database import vector_database

# doc_reader = reader(document_type="pdf", verbose=True)

# document_path = "testpdf.pdf"

# try:
#     document_content, document_name, document_type = doc_reader.load_document(document_path)
#     print(f"Document Name: {document_name}")
#     print(f"Document Type: {document_type}")
#     print(f"Document Content: {document_content}")

#     doc_adapter = adapter(chunk_size=500, chunk_overlap=10, verbose=True)

#     chunks = doc_adapter.get_chunks(document_content)

#     for i, chunk in enumerate(chunks):
#         print(f"Chunk {i + 1}:\n{chunk}\n")
    
# except Exception as e:
#     print(f"Error: {e}")

# doc_embedder = embedder(embedding_model="all-MiniLM-l6-v2", verbose=True)

# try:
#     embeddings = doc_embedder.embed_chunks(chunks)
#     print(f"Embeddings: {embeddings}")

#     query = "What is the main idea of the document?"
#     query_embedding = doc_embedder.embed_query(query)
#     print(f"Embedding for Query '{query}': {query_embedding}")

#     vdb = vector_database(verbose=True)

#     metadata = {"document_name": document_name, "document_type": document_type}
#     vdb.insert_embeddings(embeddings, chunks, metadata)

#     search_results = vdb.search_query(query_embedding, chunk_count=5)
#     print(f"Search Results: {search_results}")

# except ValueError as e:
#     print(f"Error: {e}")


import os
import sys
from reader.reader import reader
from adapter.adapter import adapter
from embedder.embedder import embedder
from vector_database.vector_database import vector_database
from llm.llm import llm

doc_reader = reader(document_type="pdf", verbose=True)
document_path = "testpdf.pdf"

try:
    document_content, document_name, document_type = doc_reader.load_document(document_path)
    print(f"Document Name: {document_name}")
    print(f"Document Type: {document_type}")
    print(f"Document Content: {document_content}")

    doc_adapter = adapter(chunk_size=500, chunk_overlap=10, verbose=True)
    chunks = doc_adapter.get_chunks(document_content)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")

    doc_embedder = embedder(embedding_model="all-MiniLM-l6-v2", verbose=True)
    embeddings = doc_embedder.embed_chunks(chunks)
    print(f"Embeddings: {embeddings}")

    vdb = vector_database(verbose=True)
    metadata = {"document_name": document_name, "document_type": document_type}
    vdb.insert_embeddings(embeddings, chunks, metadata)

    llm_module = llm(model_name="GPT4All")
    query = "What is the name of the document?"
    llm_response = llm_module.query(query, chunks)
    print(f"LLM Response: {llm_response}")

    llm_query_embedding = doc_embedder.embed_query(llm_response)
    search_results = vdb.search_query(llm_query_embedding, chunk_count=5)
    print(f"Search Results using LLM: {search_results}")

except ValueError as e:
    print(f"Error: {e}")
