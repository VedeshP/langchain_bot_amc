import os
# import streamlit as st
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def load_data():
    if not os.path.exists("./data"):
        os.makedirs("./data")

    files = os.listdir("./data")
    pdf_files = [f for f in files if f.endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the data directory.")

    # Load PDF document
    pdf_path = os.path.join("./data", pdf_files[0])
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()    

    embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY,
                credentials=None
            )
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store

# Load the vector store
vector_store = load_data()

# Set up the conversational retrieval chain
memory = ConversationBufferMemory(
    # memory_key="chat_history",
    output_key="answer",  # Specify which key to store
    return_messages=True
)

# Set up the conversational retrieval chain
chat_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
    credentials=None,
    convert_system_message_to_human=True
)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    chain_type="stuff",
    verbose=True
)


try:
    question = "What is machine learning?"
    response = retrieval_chain({
                        "question": question,
                    })
    print(response)
except Exception as e:
    print(e)
