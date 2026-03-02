from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from retrieval_pipeline import load_vector_store, retrieve_documents, format_context


def create_llm():
    """Create an Ollama LLM instance (locally running)"""
    llm = ChatOllama(model="gemma3:4b", base_url="http://192.168.64.1:11434")
    return llm


def build_prompt():
    """Build a RAG prompt template"""
    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question based only on the provided context. "
         "If the context does not contain enough information to answer, say so."),
        ("human",
         "Context:\n{context}\n\nQuestion: {question}")
    ])
    return template


def ask(query, db, llm, prompt_template):
    """Retrieve context and generate an answer for the given query"""
    relevant_docs = retrieve_documents(db, query)
    context = format_context(relevant_docs)

    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": query})
    return response.content


def main():
    db = load_vector_store()
    llm = create_llm()
    prompt_template = build_prompt()

    query = "How much did Microsoft pay to acquire Github?"

    print(f"Question: {query}\n")
    answer = ask(query, db, llm, prompt_template)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
