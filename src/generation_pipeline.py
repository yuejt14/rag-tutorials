from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from retrieval_pipeline import load_vector_store, retrieve_documents, format_context
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_NUM_CTX

MAX_HISTORY_TOKENS = int(OLLAMA_NUM_CTX * 0.3)


def create_llm():
    """Create an Ollama LLM instance (locally running)"""
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


def estimate_tokens(messages):
    """Rough token estimate: ~4 characters per token"""
    return sum(len(m.content) for m in messages) // 4


def consolidate_history(history, llm, token_count):
    """Summarize chat history into a single message to reclaim context space"""
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history
    )
    prompt = (
        "Summarize the following conversation concisely, preserving all key facts:\n\n"
        + history_text
    )
    summary = llm.invoke(prompt).content
    print(f"\n[History consolidated — was {token_count} tokens, now summarized]\n")
    return [SystemMessage(content=f"Summary of earlier conversation: {summary}")]


def build_prompt():
    """Build a RAG prompt template with chat history support"""
    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question based only on the provided context. "
         "If the context does not contain enough information to answer, say so."),
        MessagesPlaceholder("chat_history"),
        ("human",
         "Context:\n{context}\n\nQuestion: {question}")
    ])
    return template


def ask(query, db, llm, prompt_template, chat_history=None):
    """Retrieve context and generate an answer for the given query"""
    if chat_history is None:
        chat_history = []

    token_count = estimate_tokens(chat_history)
    if token_count > MAX_HISTORY_TOKENS:
        chat_history[:] = consolidate_history(chat_history, llm, token_count)

    relevant_docs = retrieve_documents(db, query)
    context = format_context(relevant_docs)

    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": query, "chat_history": chat_history})
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
