import os
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# --- Qdrant Configuration ---
QDRANT_URL = "https://ca81e711-1d1f-40a8-932c-9a2f3ca568eb.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my_excel_collection"

# --- Data and Embedding Configuration ---
# Make file path configurable - you can set this as an environment variable
EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH", r"C:\Users\Naveen\Downloads\Copy of LeadWalnut- Knowledge Base.xlsx")
EMBEDDING_MODEL = "models/embedding-001"

# --- Sheet Configuration ---
# Set to None to include all sheets, or specify a list like ["Sheet1", "Sheet2"]
SPECIFIC_SHEETS = None  # or ["Sheet1", "Data", "FAQ"] for example
EXCLUDE_SHEETS = ["Summary", "Index"]  # Sheets to exclude (optional)

def clean_text_for_embedding(text):
    """Clean and normalize text for better embedding quality."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with embeddings
    text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\{\}]', '', text)
    
    # Ensure minimum length for meaningful embeddings
    if len(text) < 10:
        return ""
    
    return text

def preview_sheets(excel_file_path):
    """Preview all sheets in the Excel file."""
    print("📋 EXCEL FILE ANALYSIS")
    print("=" * 50)
    
    try:
        excel_file = pd.ExcelFile(excel_file_path)
        print(f"File: {excel_file_path}")
        print(f"Total sheets found: {len(excel_file.sheet_names)}")
        print("\nSheet Details:")
        
        for i, sheet_name in enumerate(excel_file.sheet_names, 1):
            try:
                df = pd.read_excel(excel_file_path, sheet_name=sheet_name, nrows=0)  # Just get columns
                row_count = len(pd.read_excel(excel_file_path, sheet_name=sheet_name))
                col_count = len(df.columns)
                print(f"  {i}. '{sheet_name}' - {row_count} rows, {col_count} columns")
                
                # Show first few column names
                if col_count > 0:
                    cols_preview = list(df.columns)[:3]
                    if col_count > 3:
                        cols_preview.append("...")
                    print(f"     Columns: {cols_preview}")
                    
            except Exception as e:
                print(f"  {i}. '{sheet_name}' - Error reading: {str(e)}")
        
        print("\n" + "=" * 50)
        return excel_file.sheet_names
        
    except Exception as e:
        print(f"Error analyzing Excel file: {str(e)}")
        return []

def process_sheet_data(df, sheet_name, combination_method="smart_combined"):
    """Process data from a single sheet with improved text processing for embeddings."""
    
    if df.empty:
        return [], []
    
    # Clean the dataframe
    df = df.fillna('')
    df = df.replace('nan', '')
    df = df.replace('None', '')
    
    documents = []
    
    if combination_method == "smart_combined":
        # Improved method: combine columns intelligently for better embeddings
        sheet_documents = []
        for _, row in df.iterrows():
            row_parts = []
            for col_name, value in row.items():
                if str(value).strip() and str(value).strip() != '':
                    # Clean the value and add meaningful context
                    clean_value = clean_text_for_embedding(str(value))
                    if clean_value:
                        row_parts.append(f"{col_name}: {clean_value}")
            
            if row_parts:
                # Create a more natural sentence structure for embeddings
                combined_text = ". ".join(row_parts) + "."
                sheet_documents.append(combined_text)
    
    elif combination_method == "pipe_separated":
        # Original method: combine all columns with pipe separator
        sheet_documents = df.astype(str).agg(' | '.join, axis=1).tolist()
    
    elif combination_method == "column_labeled":
        # Label each column with its header
        sheet_documents = []
        for _, row in df.iterrows():
            row_parts = []
            for col_name, value in row.items():
                if str(value).strip() and str(value).strip() != '':
                    row_parts.append(f"{col_name}: {str(value).strip()}")
            if row_parts:
                sheet_documents.append(" | ".join(row_parts))
            else:
                sheet_documents.append("")
    
    elif combination_method == "paragraph_style":
        # Format as readable paragraphs
        sheet_documents = []
        for _, row in df.iterrows():
            parts = [str(val).strip() for val in row.values if str(val).strip() and str(val).strip() != '']
            if parts:
                sheet_documents.append(". ".join(parts) + ".")
            else:
                sheet_documents.append("")
    
    # Filter and enhance documents with better text cleaning
    valid_documents = []
    for i, doc in enumerate(sheet_documents):
        # Clean the document text
        clean_doc = clean_text_for_embedding(doc)
        
        if clean_doc and len(clean_doc) > 20:  # Minimum 20 characters for meaningful embeddings
            # Add sheet context but keep the main text clean
            enhanced_doc = f"[Sheet: {sheet_name}] {clean_doc}"
            valid_documents.append(enhanced_doc)
    
    return valid_documents

def refresh_vector_store():
    """Refresh the vector store by deleting and recreating the collection."""
    print("🔄 REFRESHING VECTOR STORE...")
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Delete existing collection
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"✅ Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            print(f"ℹ️ Collection '{COLLECTION_NAME}' did not exist.")
        
        print("✅ Vector store refreshed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error refreshing vector store: {str(e)}")
        return False

def main():
    """Enhanced multi-sheet Excel to Qdrant upload with improved embeddings."""
    
    print("🚀 MULTI-SHEET EXCEL TO QDRANT UPLOADER")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"❌ Excel file not found: {EXCEL_FILE_PATH}")
        print("Please set the EXCEL_FILE_PATH environment variable or update the code.")
        return
    
    # 1. Preview the Excel file
    available_sheets = preview_sheets(EXCEL_FILE_PATH)
    if not available_sheets:
        return
    
    # 2. Determine which sheets to process
    if SPECIFIC_SHEETS:
        sheets_to_process = [s for s in SPECIFIC_SHEETS if s in available_sheets]
        print(f"\n📌 Processing SPECIFIC sheets: {sheets_to_process}")
    else:
        sheets_to_process = [s for s in available_sheets if s not in (EXCLUDE_SHEETS or [])]
        print(f"\n📌 Processing ALL sheets except: {EXCLUDE_SHEETS or 'none'}")
        print(f"Sheets to process: {sheets_to_process}")
    
    if not sheets_to_process:
        print("❌ No sheets to process!")
        return
    
    # Ask user for confirmation
    proceed = input(f"\nProceed with uploading {len(sheets_to_process)} sheets? (y/N): ").lower().strip()
    if proceed != 'y':
        print("Upload cancelled.")
        return
    
    # 3. Initialize Qdrant and embeddings
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as e:
        print(f"❌ Error initializing Qdrant or embeddings: {str(e)}")
        return
    
    # 4. Delete and recreate collection
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' did not exist.")
    
    # 5. Process all sheets and collect documents
    all_documents = []
    all_ids = []
    document_counter = 0
    sheet_stats = {}
    
    print(f"\n📚 PROCESSING SHEETS...")
    print("-" * 40)
    
    for sheet_name in sheets_to_process:
        print(f"Processing '{sheet_name}'...")
        
        try:
            df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name)
            
            # Use improved text processing method
            valid_docs = process_sheet_data(df, sheet_name, combination_method="smart_combined")
            
            # Add to master list
            for doc in valid_docs:
                all_documents.append(doc)
                all_ids.append(document_counter)
                document_counter += 1
            
            sheet_stats[sheet_name] = {
                'total_rows': len(df),
                'valid_docs': len(valid_docs),
                'columns': list(df.columns)[:5]  # First 5 columns
            }
            
            print(f"  ✅ {len(valid_docs)} documents from '{sheet_name}' ({len(df)} total rows)")
            
        except Exception as e:
            print(f"  ❌ Error processing '{sheet_name}': {str(e)}")
            sheet_stats[sheet_name] = {'error': str(e)}
    
    if not all_documents:
        print("❌ No valid documents found!")
        return
    
    # 6. Create collection with proper vector size
    print(f"\n🔧 SETTING UP VECTOR DATABASE...")
    try:
        sample_embedding = embeddings.embed_query(all_documents[0])
        vector_size = len(sample_embedding)
        print(f"Vector dimension: {vector_size}")
    except Exception as e:
        print(f"❌ Error generating sample embedding: {str(e)}")
        return
    
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
        )
        print(f"✅ Created collection '{COLLECTION_NAME}' with {vector_size} dimensions")
    except Exception as e:
        print(f"❌ Error creating collection: {str(e)}")
        return
    
    # 7. Generate embeddings and upload in batches
    print(f"🧠 GENERATING EMBEDDINGS...")
    batch_size = 50  # Process in batches to avoid memory issues
    
    for i in range(0, len(all_documents), batch_size):
        batch_docs = all_documents[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        
        print(f"  Processing batch {i//batch_size + 1}/{(len(all_documents) + batch_size - 1)//batch_size}")
        
        try:
            # Generate embeddings for this batch
            batch_embeddings = embeddings.embed_documents(batch_docs)
            
            # Prepare points for this batch
            points = []
            for j, doc_text in enumerate(batch_docs):
                doc_id = batch_ids[j]
                
                # Extract sheet info
                sheet_name = "Unknown"
                if doc_text.startswith('[Sheet: '):
                    end_bracket = doc_text.find(']')
                    if end_bracket != -1:
                        sheet_name = doc_text[8:end_bracket]
                        original_text = doc_text[end_bracket + 2:]
                    else:
                        original_text = doc_text
                else:
                    original_text = doc_text
                
                metadata = {
                    "sheet_name": sheet_name,
                    "document_id": doc_id,
                    "char_count": len(original_text),
                    "batch_id": i//batch_size + 1
                }
                
                points.append(models.PointStruct(
                    id=doc_id,
                    vector=batch_embeddings[j],
                    payload={
                        "text": original_text,
                        "enhanced_text": doc_text,
                        "metadata": metadata
                    }
                ))
            
            # Upload batch to Qdrant
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"    ✅ Uploaded batch with {len(points)} documents")
            
        except Exception as e:
            print(f"    ❌ Error processing batch: {str(e)}")
            continue
    
    # 8. Final summary
    print(f"\n🎉 UPLOAD COMPLETE!")
    print("=" * 50)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total documents: {len(all_documents)}")
    print(f"Vector dimension: {vector_size}")
    
    print(f"\n📊 SHEET BREAKDOWN:")
    for sheet_name, stats in sheet_stats.items():
        if 'error' in stats:
            print(f"  ❌ {sheet_name}: {stats['error']}")
        else:
            print(f"  ✅ {sheet_name}: {stats['valid_docs']} docs from {stats['total_rows']} rows")
    
    # Verify upload
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"\n✅ Verification: {collection_info.points_count} points in collection")
    except Exception as e:
        print(f"\n⚠️ Could not verify collection: {str(e)}")

if __name__ == "__main__":
    # First refresh the vector store
    if refresh_vector_store():
        # Then run the main upload process
        main()
    else:
        print("❌ Failed to refresh vector store. Exiting.")