
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("CivGPT – Ask Your Council Anything")

query = st.text_input("Ask a question (e.g., 'How do I register my cat?')")

if query:
    with st.spinner("CivGPT is checking council docs..."):
        loader = PyPDFLoader("wyndham_pet_rego.pdf")
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        embeddings = OpenAIEmbeddings()
from chromadb.config import Settings

vectordb = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="./chroma",
    client_settings=Settings(anonymized_telemetry=False, persist_directory="./chroma")
)
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4"),
            chain_type="stuff",
            retriever=retriever
        )
        answer = qa_chain.run(query)
        st.success(answer)
