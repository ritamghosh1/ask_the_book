import os
import chromadb # type: ignore
from llama_index.core import Document, VectorStoreIndex, StorageContext # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # type: ignore
from llama_index.core.node_parser import SemanticSplitterNodeParser # type: ignore
from llama_index.vector_stores.chroma import ChromaVectorStore # type: ignore

# Importing the document processor
from ingestion import DocumentProcessor

class VectorDBManager:
    def __init__(self, collection_name = "ask_the_book"):
        self.collection_name = collection_name

        # Downloading the BGE-small embedding model
        print("Loading Embedding Model (BAAI/bge-small-en-v1.5)...")
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Setup Chromadb with Ephemeral Storage
        print("Initializing In-Memory ChromaDB...")
        self.db_client = chromadb.EphemeralClient()
        self.chroma_collection = self.db_client.get_or_create_collection(name=self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def process_and_store(self, pdf_path: str):
        # Clear any existing document from our storage
        if self.chroma_collection.count()>0:
            print("Clearing old document from Memory...")
            self.db_client.delete_collection(name=self.collection_name)
            self.chroma_collection = self.db_client.create_collection(name=self.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        print("Starting Ingestion Pipeline...")
        processor = DocumentProcessor(pdf_path)
        pages = processor.process()

        if not pages:
            print("No text extracted. Exiting.")
            return
        
        # Converting our dictionaries to LLAMA docs
        llama_docs = []
        for page in pages:
            llama_docs.append(Document(
                text=page["content"],
                metadata={"page": page["page_num"], "source": pdf_path}
            ))

        print(f"Loaded {len(llama_docs)} pages into memory. Starting Semantic Chunking...")

        # Semantic Chunking
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=85, 
            embed_model=self.embed_model
        )
        nodes = splitter.get_nodes_from_documents(llama_docs)
        print(f"Created {len(nodes)} semantic chunks, Saving to ChromaDB RAM...")
        index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
                
        print("Success! Database is currently held in RAM.")
        return index,nodes
                    
if __name__ == "__main__":
    pdf_path = "Sample_book.pdf"
    if os.path.exists(pdf_path):
        manager = VectorDBManager()
        manager.process_and_store(pdf_path)
    else:
        print(f"Could not find {pdf_path}")