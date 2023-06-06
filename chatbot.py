from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from langchain.llms import AzureOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain


def make_chain():
    model = AzureOpenAI(
        deployment_name="chat",
        model_name='gpt-35-turbo'
    )

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

    vector_store = Chroma(persist_directory='./data/chroma', embedding_function=embeddings)

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )


if __name__ == '__main__':
    load_dotenv()

    chain = make_chain()
    chat_history = []

    while True:
        print()
        question = input("Enter a question: ")

        response = chain({"question": question, "chat_history": chat_history})

        answer = response["answer"]
        source = response["source_documents"]
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        print("/n/nSources:\n")
        for document in source:
            print(f"Page: {document.metadata['page_number']}")
            print(f"Text: {document.page_content}...\n")
        print(f"Answer: {answer}")
