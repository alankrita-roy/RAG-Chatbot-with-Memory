import os
import re
import uuid
import streamlit as st

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Load secrets (for cloud)
HF_TOKEN = st.secrets["HF_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["HF_TOKEN"] = HF_TOKEN

# Initialize LLM and embeddings
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("Conversational RAG with PDF Uploads (Streamlit Cloud)")
st.write("Upload PDFs and ask questions about their content.")

# Session ID management
session_id = st.text_input("Session ID", value="default_session")
if "store" not in st.session_state:
    st.session_state.store = {}

# File helper: safely save uploaded file
def save_uploaded_file(uploaded_file):
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', uploaded_file.name)
    unique_filename = f"{uuid.uuid4()}_{safe_name}"
    save_path = os.path.join(".", unique_filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return save_path

uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    saved_file_paths = []

    # Save and load PDFs
    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        saved_file_paths.append(file_path)
        loader = PyPDFLoader(file_path)
        all_docs.extend(loader.load())

    # Split and embed documents 
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_documents(all_docs)
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # History-aware question
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference previous context, "
                   "reformulate a standalone question. Do not answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant answering questions using retrieved documents. "
                   "Be concise (max 3 sentences). If unsure, say 'I don't know'.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


    # Chat history state
    def get_chat_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Input & Output
    user_query = st.text_input("Your question:")
    if user_query:
        chat_history = get_chat_history(session_id)
        result = conversational_chain.invoke(
            {"input": user_query},
            config={"configurable": {"session_id": session_id}}
        )
        st.markdown("**Assistant:** " + result["answer"])
        st.markdown("**Chat History:**")
        for msg in chat_history.messages:
            role = "🧑 You" if msg.type == "human" else "🤖 Assistant"
            st.markdown(f"**{role}:** {msg.content}")

    # Clean up files after processing
    for path in saved_file_paths:
        if os.path.exists(path):
            os.remove(path)
