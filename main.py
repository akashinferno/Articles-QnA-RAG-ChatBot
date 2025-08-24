import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

embeddings =GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    async_client=False
    )

st.title("News Research RAG QnA bot")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

FILE_PATH ="vectordb.pkl"

if process_url_clicked:

    # Load and split the documents
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data  = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        )
    main_placeholder.text("Data Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and vector store

    vectordb = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding vector Started Building...✅✅✅")
    time.sleep(2)
    # with open(FILE_PATH, "wb") as f:
    #     pickle.dump(vectordb, f)
    vectordb.save_local("faiss_index")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)