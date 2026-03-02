from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, CHROMA_DIR, CHROMA_METADATA


def load_vector_store(persist_directory=CHROMA_DIR):
    """Load the persisted ChromaDB vector store"""
    print("Loading vector store...")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata=CHROMA_METADATA,
    )

    print(f"Vector store loaded from {persist_directory}")
    return db


def retrieve_documents(db, query, k=3):
    """Retrieve relevant documents for a given query"""
    print(f"Retrieving documents for: {query}")

    relevant_docs = db.similarity_search(query, k=k)

    print(f"Found {len(relevant_docs)} relevant documents")
    return relevant_docs


def format_context(documents):
    """Format retrieved documents into a single context string"""
    return "\n\n".join(doc.page_content for doc in documents)


def main():
    query = "How much did Microsoft pay to acquire Github?"

    db = load_vector_store()
    relevant_docs = retrieve_documents(db, query)

    print(f"\nQuery: {query}")
    print("--------- Context ---------")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i}:\n{doc.page_content}\n")


if __name__ == "__main__":
    main()
