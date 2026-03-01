from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persistence_dir = "db/chroma_db"

# load embedding and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistence_dir,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# search for relevent documents
query = "In 1999, where did Google move its headquarters to?"

retriever = db.as_retriever(search_kwargs={"k": 3})

relevant_docs = retriever.invoke(query)

print(f"Query: {query}")
# Display result
print("--------- Context ---------")
for i, doc in enumerate(relevant_docs):
    print(f"---- Document {i+1} ----")
    print(f"    Source: {doc.metadata['source']}")
    print(f"    Length: {len(doc.page_content)} chars") 