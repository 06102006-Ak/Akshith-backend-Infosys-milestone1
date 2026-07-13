import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# Using a clean, alternative embedding class that does not import torch
from langchain_community.embeddings import DeterministicFakeEmbedding

class DocumentAgent:
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        # Temporary deterministic vector generator to bypass PyTorch completely
        # For production, we will hook this to a free HuggingFace API endpoint via requests
        self.embeddings = DeterministicFakeEmbedding(size=384)

    def extract_text_with_page_metadata(self, pdf_path: str):
        """Extracts text while maintaining strict page boundaries for citations"""
        doc = fitz.open(pdf_path)
        pages = []
        for page_idx, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "text": text,
                    "page_number": page_idx + 1
                })
        return pages

    def parse_and_index(self, pdf_path: str, company_name: str, session_id: str):
        """Chunks the text and saves it into an isolated vector database collection"""
        pages = self.extract_text_with_page_metadata(pdf_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        
        chunks = []
        metadatas = []
        
        for page in pages:
            splits = text_splitter.split_text(page["text"])
            for split in splits:
                chunks.append(split)
                metadatas.append({
                    "session_id": session_id,
                    "company_name": company_name.upper(),
                    "page_number": page["page_number"]
                })
        
        Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.db_path,
            collection_name=f"session_{session_id}"
        )
        return len(chunks)