---

# Multi-Document RAG Chatbot with Groq & Streamlit

This repository contains a powerful and flexible Retrieval-Augmented Generation (RAG) system that allows you to chat with your documents. It leverages the speed of the Groq API for near-instant responses, Google's state-of-the-art models for embeddings, and Streamlit for a user-friendly interface.

The project offers two distinct modes of operation:
1.  **Dynamic File Chat (`app.py`):** Upload various document formats (PDF, CSV, DOCX, TXT, XLSX) on-the-fly, and start a conversation immediately. This mode uses an in-memory FAISS vector store, perfect for temporary and quick analysis.
2.  **Persistent Knowledge Base Chat (`app1.py` & `qdrant_store.py`):** Process a large, structured Excel file and store it in a persistent Qdrant cloud database. The corresponding Streamlit app then connects to this knowledge base for robust, long-term use.

## 🚀 Features

- **Multi-Format Support:** Natively chat with PDF, CSV, Excel, DOCX, and TXT files.
- **Blazing Fast LLM:** Powered by the Groq API for extremely low-latency conversations.
- **High-Quality Embeddings:** Uses Google's `embedding-001` model for accurate document understanding.
- **Dual Vector Store Options:**
    - **FAISS:** For fast, in-memory vector storage for uploaded files.
    - **Qdrant:** For a robust, persistent, and scalable cloud-based vector store.
- **Intelligent Data Processing:** The Qdrant ingestion script cleans and structures data from Excel sheets for optimal retrieval.
- **Conversation History:** Remembers previous questions and answers in a session.
- **Source Citing:** Displays the source documents that were used to generate an answer, providing transparency and trust.
- **User-Friendly Interface:** Built with Streamlit for a clean and interactive experience.

## 🛠️ Prerequisites

Before you begin, ensure you have the following:
- **Python 3.8+**
- **API Keys:**
    - **Google API Key:** For generating embeddings.
    - **Groq API Key:** For accessing the LLM.
    - **Qdrant API Key & URL:** (Required only for the persistent knowledge base app).

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create and configure your environment file:**
    - Make a copy of `config_template.txt` and rename it to `.env`.
    - Open the `.env` file and add your API keys and file paths.

    ```dotenv
    # .env file

    # Required for both applications
    GOOGLE_API_KEY="your_google_api_key"
    GROQ_API_KEY="your_groq_api_key"

    # --- Required ONLY for the Persistent Knowledge Base App (app1.py & qdrant_store.py) ---

    # Qdrant credentials
    QDRANT_API_KEY="your_qdrant_api_key"
    QDRANT_URL="your_qdrant_cloud_url"

    # Absolute path to the Excel file you want to ingest into Qdrant
    EXCEL_FILE_PATH="C:/path/to/your/knowledge_base.xlsx"
    ```

## 🚀 Usage

This project offers two independent applications. Choose the one that fits your needs.

---

### Option 1: Chat with Uploaded Files (Local FAISS Store)

This application allows you to upload documents directly in the UI and start asking questions. The data is processed in-memory and is cleared when the app is closed.

**How to Run:**
```bash
streamlit run app.py
```

**Workflow:**
1.  Launch the app.
2.  Use the sidebar to upload one or more files (`pdf`, `csv`, `xlsx`, `docx`, `txt`).
3.  Click the "Process Documents" button.
4.  Wait for the files to be loaded, chunked, and embedded.
5.  Start asking questions in the chat input at the bottom!

---

### Option 2: Chat with a Persistent Knowledge Base (Qdrant)

This is a two-step process designed for a more permanent and robust knowledge base built from a large Excel file.

#### Step 1: Ingest Your Data into Qdrant

First, you need to run the `qdrant_store.py` script. This script will read your specified Excel file, process its sheets, generate embeddings, and upload them to your Qdrant collection.

**Important:** Make sure `EXCEL_FILE_PATH`, `QDRANT_API_KEY`, and `QDRANT_URL` are correctly set in your `.env` file.

**How to Run:**
```bash
python qdrant_store.py
```
This script will:
1.  Connect to your Qdrant instance.
2.  Delete the existing collection to ensure a fresh start.
3.  Read and intelligently parse the data from each sheet in your Excel file.
4.  Generate embeddings for the data in batches.
5.  Upload the embeddings and metadata (like sheet name) to your Qdrant collection.

#### Step 2: Run the Chat Application

Once your data is in Qdrant, you can start the chat interface that connects to it.

**How to Run:**
```bash
streamlit run app1.py
```

**Workflow:**
1.  The app will automatically connect to the Qdrant collection specified in the script.
2.  The chat interface will load, showing it's connected to the knowledge base.
3.  Start asking questions! The app will retrieve relevant context from your Qdrant database to provide accurate answers.

## 📁 File Structure

```
.
├── app.py                  # Streamlit App: Chat with uploaded files (FAISS)
├── app1.py                 # Streamlit App: Chat with persistent knowledge base (Qdrant)
├── qdrant_store.py         # Script to ingest Excel data into Qdrant
├── requirements.txt        # Python dependencies
├── htmltemplate.py         # (Optional) HTML/CSS templates for UI styling
├── config_template.txt     # Template for your .env file
└── README.md               # This file
```

## 🔧 Troubleshooting

- **`GOOGLE_API_KEY not found`**: Ensure your `.env` file is created correctly and contains your valid Google API Key.
- **`Qdrant connection error`**: Double-check your `QDRANT_URL` and `QDRANT_API_KEY` in the `.env` file. Ensure your Qdrant instance is active.
- **`Excel file not found`**: Make sure the `EXCEL_FILE_PATH` in your `.env` file is an absolute path and that the file exists.
- **`unstructured` library errors**: The `unstructured` library is powerful but can have complex dependencies. If you encounter issues, especially with Excel files, try running `pip install "unstructured[xlsx]"`.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
