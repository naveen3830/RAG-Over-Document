# ChatWithPdf's - Document Embedding System

This project provides a robust system for embedding Excel documents into a Qdrant vector database using Google's Generative AI embeddings.

## 🚀 What's New (Fixed Issues)

The document embedding system has been significantly improved to address several issues:

### ✅ **Fixed Problems:**
1. **Missing Dependencies** - Added `qdrant-client`, `pandas`, and `python-dotenv` to requirements.txt
2. **Better Text Processing** - Improved text cleaning and normalization for better embedding quality
3. **Batch Processing** - Documents are now processed in batches to avoid memory issues
4. **Error Handling** - Better error handling and user feedback throughout the process
5. **Configurable File Paths** - Excel file path can now be set via environment variables
6. **Text Quality** - Better text cleaning removes special characters that interfere with embeddings

### 🔧 **Key Improvements:**
- **Smart Text Combination**: Documents are now combined using natural sentence structure instead of pipe separators
- **Batch Processing**: Large documents are processed in batches of 50 for better memory management
- **Text Cleaning**: Special characters and extra whitespace are properly cleaned
- **Better Metadata**: Enhanced metadata tracking including batch information
- **Error Recovery**: Continues processing even if individual batches fail

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Google API Key** for embeddings
3. **Qdrant API Key** for vector database
4. **Excel file** to process

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ChatWithPdf's
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file:**
   ```bash
   # Copy the template
   cp config_template.txt .env
   
   # Edit .env with your actual API keys
   GOOGLE_API_KEY=your_actual_google_api_key
   QDRANT_API_KEY=your_actual_qdrant_api_key
   EXCEL_FILE_PATH=C:\path\to\your\excel\file.xlsx
   ```

## 🧪 Testing

Before running the main process, test your setup:

```bash
python test_embeddings.py
```

This will verify:
- ✅ Google API key and embedding generation
- ✅ Qdrant connection
- ✅ Excel file accessibility

## 📤 Usage

### Option 1: Refresh + Upload (Recommended)
```bash
python refresh_and_upload.py
```
This script will:
1. 🔄 Refresh the vector store (delete existing collection)
2. 📤 Upload documents with new embeddings

### Option 2: Direct Upload
```bash
python qdrant_store.py
```
This will run the main upload process directly.

## 🔍 How It Works

1. **Document Processing**: Excel sheets are read and processed using intelligent text combination
2. **Text Cleaning**: Special characters and formatting are cleaned for better embedding quality
3. **Embedding Generation**: Google's embedding-001 model creates vector representations
4. **Batch Upload**: Documents are uploaded to Qdrant in batches of 50
5. **Metadata Tracking**: Each document includes sheet name, ID, and character count

## 📊 Configuration Options

### Excel Processing
- **Combination Methods**: Choose how columns are combined:
  - `smart_combined` (default): Natural sentence structure
  - `pipe_separated`: Pipe-separated values
  - `column_labeled`: Column name + value pairs
  - `paragraph_style`: Paragraph format

### Sheet Selection
- **Include All**: Process all sheets except excluded ones
- **Specific Sheets**: Process only specified sheets
- **Exclude Sheets**: Skip specific sheets (e.g., "Summary", "Index")

## 🚨 Troubleshooting

### Common Issues:

1. **"GOOGLE_API_KEY not found"**
   - Create a `.env` file with your Google API key
   - Ensure the key has access to the embedding model

2. **"QDRANT_API_KEY not found"**
   - Add your Qdrant API key to the `.env` file
   - Verify the Qdrant URL is correct

3. **"Excel file not found"**
   - Set `EXCEL_FILE_PATH` in your `.env` file
   - Use absolute paths for better reliability

4. **Memory Issues**
   - The system now processes documents in batches of 50
   - If still having issues, reduce `batch_size` in the code

### Performance Tips:
- Use SSD storage for Excel files
- Ensure stable internet connection for API calls
- Process large files during off-peak hours

## 📈 Monitoring

The system provides detailed progress information:
- Sheet-by-sheet processing status
- Document counts per sheet
- Batch upload progress
- Final verification counts

## 🔄 Refreshing Data

To completely refresh your vector store:
1. Run `python refresh_and_upload.py`
2. This will delete the existing collection and recreate it
3. All documents will be re-processed with fresh embeddings

## 📝 File Structure

```
ChatWithPdf's/
├── qdrant_store.py          # Main document processing and upload
├── refresh_and_upload.py    # Refresh + upload script
├── test_embeddings.py       # Test script for verification
├── requirements.txt          # Python dependencies
├── config_template.txt       # Environment variables template
├── README.md                # This file
└── .env                     # Your environment variables (create this)
```

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python test_embeddings.py` to identify specific problems
3. Verify your API keys and file paths
4. Check the console output for detailed error messages

## 📄 License

This project is part of the ChatWithPdf's system. Please refer to your project's license terms.
