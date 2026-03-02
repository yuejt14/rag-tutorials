from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_vector_store(persist_directory="db/chroma_db"):
    """Load the persisted ChromaDB vector store"""
    print("Loading vector store...")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store loaded from {persist_directory}")
    return db


def retrieve_documents(db, query, k=3):
    """Retrieve relevant documents for a given query"""
    print(f"Retrieving documents for: {query}")

    retriever = db.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(query)

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
