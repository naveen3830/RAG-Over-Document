import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

QDRANT_URL = "https://ca81e711-1d1f-40a8-932c-9a2f3ca568eb.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my_excel_collection"

EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "llama-3.3-70b-versatile"  # Or another model like "gemma-7b-it"

# Custom prompt template for better responses with sheet awareness
CUSTOM_PROMPT = """Use the following pieces of context to answer the question at the end. 
The context includes information from different Excel sheets - pay attention to which sheet each piece of information comes from.
If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.
Always provide specific details from the context when available, and mention which sheet the information comes from when relevant.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""

@st.cache_resource
def get_vector_store():
    """Initializes and returns a connection to the Qdrant vector store."""
    try:
        # Initialize the Qdrant client
        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            timeout=30
        )
        
        # Test connection
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            st.error(f"Collection '{COLLECTION_NAME}' not found. Available collections: {collection_names}")
            st.stop()
        
        # Initialize the embedding model
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create the Qdrant vector store object with content filter
        vector_store = Qdrant(
            client=client, 
            collection_name=COLLECTION_NAME, 
            embeddings=embeddings,
            content_payload_key="text",  # Specify the payload key for content
            metadata_payload_key="metadata"
        )
        
        logger.info(f"Successfully connected to Qdrant collection: {COLLECTION_NAME}")
        return vector_store
        
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {str(e)}")
        st.stop()

@st.cache_resource
def get_conversation_chain(_vector_store):
    """Creates a conversational retrieval chain."""
    try:
        llm = ChatGroq(
            temperature=0.1,  # Slightly higher for more natural responses
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name=LLM_MODEL,
            max_tokens=1024
        )
        
        # Custom prompt template
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT,
            input_variables=["context", "chat_history", "question"]
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True, 
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,  # Retrieve more docs to account for potential None values
                    "filter": None  # We'll handle filtering in post-processing
                }
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        logger.info("Successfully created conversation chain")
        return conversation_chain
        
    except Exception as e:
        st.error(f"Failed to create conversation chain: {str(e)}")
        st.stop()

def display_source_documents(source_docs):
    """Display source documents in a formatted way with sheet information."""
    if source_docs:
        st.subheader("📚 Source Information")
        valid_docs = []
        for doc in source_docs:
            # Filter out documents with None or empty content
            if doc and hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                valid_docs.append(doc)
        
        if not valid_docs:
            st.write("No valid source documents found.")
            return
        
        # Group documents by sheet
        sheet_groups = {}
        for doc in valid_docs:
            sheet_name = "Unknown Sheet"
            if hasattr(doc, 'metadata') and doc.metadata and 'sheet_name' in doc.metadata:
                sheet_name = doc.metadata['sheet_name']
            elif doc.page_content.startswith('[Sheet: '):
                end_bracket = doc.page_content.find(']')
                if end_bracket != -1:
                    sheet_name = doc.page_content[8:end_bracket]
            
            if sheet_name not in sheet_groups:
                sheet_groups[sheet_name] = []
            sheet_groups[sheet_name].append(doc)
        
        # Display grouped by sheet
        for sheet_name, docs in sheet_groups.items():
            st.markdown(f"### 📋 From Sheet: **{sheet_name}**")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Source {i} from {sheet_name}"):
                    content = doc.page_content.strip()
                    
                    # Remove sheet prefix if present for cleaner display
                    if content.startswith('[Sheet: '):
                        end_bracket = content.find(']')
                        if end_bracket != -1:
                            content = content[end_bracket + 2:].strip()
                    
                    if len(content) > 500:
                        st.markdown(f"**Content:** {content[:500]}...")
                    else:
                        st.markdown(f"**Content:** {content}")
                        
                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.markdown(f"**Metadata:** {doc.metadata}")

def handle_user_input(user_question):
    """Processes user input and generates response."""
    try:
        with st.spinner("🤔 Thinking..."):
            # Get response from the conversation chain
            response = st.session_state.conversation.invoke({"question": user_question})
            
            # Add messages to chat history
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response['answer']))
            
            # Display the new response
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                
                # Display source documents
                if 'source_documents' in response and response['source_documents']:
                    display_source_documents(response['source_documents'])
            
            return response
            
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        logger.error(f"Error in handle_user_input: {str(e)}")
        return None

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []
    st.session_state.conversation.memory.clear()
    st.rerun()

def main():
    st.set_page_config(
        page_title="Knowledge Base Chat", 
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Knowledge Base Chat Assistant")
    st.markdown("Ask questions about your uploaded data and get intelligent responses!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Display connection info
        st.info(f"**Collection:** {COLLECTION_NAME}")
        st.info(f"**Model:** {LLM_MODEL}")
        
        # Show sheet-specific search options
        st.subheader("🔍 Search Options")
        search_specific_sheet = st.selectbox(
            "Filter by Sheet (Optional)",
            options=["All Sheets", "Ask me to detect sheets first"],
            help="Choose to search in all sheets or ask about available sheets first"
        )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_chat_history()
        
        # Display retrieval settings
        st.subheader("📊 Retrieval Settings")
        st.write("- **Search Type:** Similarity")
        st.write("- **Retrieved Docs:** Top 5")
        st.write("- **Temperature:** 0.1")
        st.write("- **Multi-Sheet:** Enabled")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize the conversation chain
    if st.session_state.conversation is None:
        with st.spinner("🚀 Initializing knowledge base connection..."):
            try:
                vector_store = get_vector_store()
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("✅ Successfully connected to knowledge base!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                st.stop()

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Handle new user input
    if user_question := st.chat_input("💭 Ask a question about your knowledge base..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Process and display response
        handle_user_input(user_question)

    # Display helpful information at the bottom
    if not st.session_state.chat_history:
        st.markdown("---")


if __name__ == '__main__':
    main()