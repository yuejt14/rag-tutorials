import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def load_documents(doc_path="docs"):
    """Load all the text files from the doc directory"""
    print(f"Loading documents from {doc_path}...")

    #check if docs directory exists
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"{doc_path} does not exist")

    # Load all .text files from the doc directory
    loader = DirectoryLoader(
        doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"Not .txt files found in {doc_path}. Please add your documents.")

    for i, doc in enumerate(documents[:2]): # show first two documents
        print(f"\nDocument {i+1}:")
        print(f"    Source: {doc.metadata['source']}")
        print(f"    Content Length: {len(doc.page_content)} chars")
        print(f"    Metadata: {doc.metadata}")
        print(f"    Content Preview: {doc.page_content[:100]}...")
    return documents


def split_documents(documents, chunk_size=100, chunk_overlap=0):
    """Split documents into chunks of size chunk_size"""
    print("Splitting documents...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"---- Chunk {i+1} ----")
            print(f"    Source: {chunk.metadata['source']}")
            print(f"    Length: {len(chunk.page_content)} chars")
            print(f"    Content:")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks) > 5:
            print(f"\n and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating vector store in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Chroma Vector store
    print("---- Creating Vector Store ----")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished createing vector store ---")
    print(f"Vector store create and saved to {persist_directory}")
    return vectorstore

def main():
    print("Ingestion pipeline")

    #1. Loading the files
    documents = load_documents(doc_path="docs")
    #2. Chunking the files
    chunks = split_documents(documents)
    #3. Embedding the files
    create_vector_store(chunks)



if __name__ == "__main__":
    main()