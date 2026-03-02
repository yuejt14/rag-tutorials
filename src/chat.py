from generation_pipeline import create_llm, build_prompt, ask
from retrieval_pipeline import load_vector_store
from langchain_core.messages import HumanMessage, AIMessage


def main():
    print("Loading RAG pipeline...")
    db = load_vector_store()
    llm = create_llm()
    prompt_template = build_prompt()
    print("Ready! Type 'quit' or 'exit' to stop.\n")

    chat_history = []

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        answer = ask(query, db, llm, prompt_template, chat_history)
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
