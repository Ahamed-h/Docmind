from core.ingestion import ingest_folder
from core.config import DATA_PATH

if __name__ == "__main__":
    print("Building document index...")
    total = ingest_folder(DATA_PATH)
    print(f"\nDone. {total} chunks indexed.")
    print("Run: streamlit run streamlit_app.py")