from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, CHROMA_DIR, CHROMA_METADATA


def load_documents(doc_path="docs"):
    """Load all the text files from the doc directory"""
    print(f"Loading documents from {doc_path}...")

    # Load all .txt files from the doc directory
    loader = DirectoryLoader(
        doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"No .txt files found in {doc_path}. Please add your documents.")

    for i, doc in enumerate(documents[:2]): # show first two documents
        print(f"\nDocument {i+1}:")
        print(f"    Source: {doc.metadata['source']}")
        print(f"    Content Length: {len(doc.page_content)} chars")
        print(f"    Metadata: {doc.metadata}")
        print(f"    Content Preview: {doc.page_content[:100]}...")
    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks of size chunk_size"""
    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
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

def create_vector_store(chunks, persist_directory=CHROMA_DIR):
    """Create and persist ChromaDB vector store"""
    print("Creating vector store in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create Chroma Vector store
    print("---- Creating Vector Store ----")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata=CHROMA_METADATA,
    )

    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")
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