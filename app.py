import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import tempfile
import time

import nest_asyncio
nest_asyncio.apply()

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader
)

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage

load_dotenv()

def load_documents(uploaded_files):
    """Loads text from uploaded files using the appropriate LangChain loader."""
    documents = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tf:
            tf.write(file.getvalue())
            temp_path = tf.name

        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif file_extension == ".csv":
            loader = CSVLoader(temp_path)
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(temp_path, mode="elements")
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_path)
        else:
            st.warning(f"Unsupported file type: {file.name}. Skipped.")
            continue
            
        documents.extend(loader.load())
        os.remove(temp_path)
        
    return documents


def get_text_chunks(documents):
    """Splits a list of documents into smaller chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# --- NEW BATCHING FUNCTION TO PREVENT TIMEOUTS ---
def get_vectorstore_with_batching(text_chunks):
    """
    Creates a vector store from text chunks using batching to avoid API timeouts.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Set a batch size
    batch_size = 100
    
    # Initialize an empty FAISS vector store
    # We use the first chunk to get the dimensionality, then add the rest.
    first_chunk_embedding = embeddings.embed_query(text_chunks[0].page_content)
    index_dimension = len(first_chunk_embedding)
    
    vectorstore = FAISS.from_texts(
        texts=[text_chunks[0].page_content],
        embedding=embeddings,
        metadatas=[text_chunks[0].metadata]
    )

    # Prepare a progress bar
    progress_bar = st.progress(0, text="Embedding documents...")

    # Process the rest of the chunks in batches
    for i in range(1, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        
        # Extract page content and metadata
        batch_texts = [chunk.page_content for chunk in batch]
        batch_metadatas = [chunk.metadata for chunk in batch]
        
        # Add the batch to the vector store
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
        
        # Update progress bar
        progress_value = min((i + batch_size) / len(text_chunks), 1.0)
        progress_bar.progress(progress_value, text=f"Embedding documents... {int(progress_value * 100)}% complete")

    progress_bar.empty() # Clear the progress bar
    return vectorstore


def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain using Groq."""
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name="llama-3.3-70b-versatile"
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
    )
    return conversation_chain

def stream_handler(stream):
    for chunk in stream:
        if 'answer' in chunk:
            yield chunk['answer']
        if 'source_documents' in chunk:
            st.session_state.source_docs = chunk['source_documents']


def main():
    st.set_page_config(
        page_title="Multi-Format RAG Chat",
        page_icon="📁",
        layout="wide"
    )

    st.title("📁 Chat with Any Document")
    st.markdown("Upload PDFs, CSVs, Excel, TXT, or DOCX files and get instant answers.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "source_docs" not in st.session_state:
        st.session_state.source_docs = []

    with st.sidebar:
        st.header("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload your files here and click 'Process'",
            accept_multiple_files=True,
            type=['pdf', 'csv', 'txt', 'docx', 'xlsx'] 
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.error("Please upload at least one document.")
            else:
                # The spinner now covers the loading and chunking part
                with st.spinner("Loading and chunking documents..."):
                    try:
                        documents = load_documents(uploaded_files)
                        if not documents:
                            st.error("Could not load any documents. Please check file formats.")
                            return
                        
                        text_chunks = get_text_chunks(documents)
                        if not text_chunks:
                            st.error("Could not create text chunks from the documents.")
                            return
                    except Exception as e:
                        st.error(f"An error occurred while loading or chunking documents: {e}")
                        return
                
                # --- MODIFIED PROCESSING LOGIC ---
                # The embedding process is now outside the spinner and has its own progress bar
                try:
                    st.info("Starting the embedding process... This may take a moment for large files.")
                    vectorstore = get_vectorstore_with_batching(text_chunks)
                    
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_history = []
                    st.session_state.source_docs = []
                    st.success("Ready! Ask your questions now.")
                except Exception as e:
                    st.error(f"An error occurred during embedding: {e}")

    # Main chat interface remains the same
    for message in st.session_state.chat_history:
        with st.chat_message(name="user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

    if user_question := st.chat_input("Ask a question about your documents..."):
        if st.session_state.conversation is None:
            st.error("Please process your documents first.")
            return

        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history.append(HumanMessage(content=user_question))

        with st.spinner("Searching documents and preparing Groq's response..."):
            stream = st.session_state.conversation.stream({"question": user_question})
            with st.chat_message("assistant"):
                full_response = st.write_stream(stream_handler(stream))

        st.session_state.chat_history.append(AIMessage(content=full_response))

    if st.session_state.source_docs:
        with st.sidebar:
            with st.expander("View Retrieved Sources", expanded=False):
                for i, doc in enumerate(st.session_state.source_docs):
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    st.markdown(f"**Source {i+1}:** `{source_file}`")
                    st.info(doc.page_content)

if __name__ == "__main__":
    main()
