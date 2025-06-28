from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def ask_ai(question):
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    retriever = vectordb.as_retriever()
    llm = Ollama(model="llama3")  # Or "deepseek-coder" if you have DeepSeek
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa.run(question)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    ask_ai("What is the termination clause in the contract?")
