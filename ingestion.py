import os
import re
from dotenv import load_dotenv # type: ignore
from unstructured.partition.pdf import partition_pdf # type: ignore

# Load environment variables from .env file 
load_dotenv()

class DocumentProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.elements = []
        self.structured_pages = []

    def extract_text(self):
        """
        Upgraded Parsing: Using `unstructured` instead of PyMuPDF.
        Handles mixed layouts, tables, and automatically uses OCR for scanned pages.
        """
        print(f"Loading and partitioning document: {self.filepath}...")
        try:
            # strategy="auto" will use fast text extraction for digital PDFs, 
            # and automatically fall back to "hi_res" (OCR) if the page is an image.
            raw_elements = partition_pdf(
                filename=self.filepath,
                strategy="auto", 
            )
            
            # unstructured gives us semantic elements (Title, NarrativeText, Table, etc.)
            for el in raw_elements:
                text = str(el)
                
                # Safely grab the page number, defaulting to 1 if it is missing or None
                page_num = 1
                if hasattr(el, "metadata") and el.metadata and hasattr(el.metadata, "page_number") and el.metadata.page_number:
                    page_num = el.metadata.page_number
                
                if text.strip():
                    self.elements.append({
                        "page_num" : page_num, 
                        "text" : text,
                        "type": type(el).__name__
                    })

            print(f"Successfully extracted {len(self.elements)} elements.")
        except Exception as e:
            print(f"Error reading document : {e}")
            return []

    def clean_text(self, text: str) -> str:
        """
        Because `unstructured` is very good at omitting headers/footers automatically,
        we just need basic cleanup here compared to raw PyMuPDF.
        """
        if not text:    
            return ""
        
        # Remove excessive empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def process(self) -> list[dict]:
        """
        Runs the full ingestion pipeline and groups the semantic elements back into pages.
        """
        self.extract_text()
        
        print("Cleaning and grouping extracted elements...")
        grouped_pages = {}
        
        # Group the isolated elements back by their page number
        for item in self.elements:
            cleaned = self.clean_text(item["text"])
            page_num = item["page_num"]
            
            if cleaned:
                if page_num not in grouped_pages:
                    grouped_pages[page_num] = []

                grouped_pages[page_num].append(cleaned)
            
        for page_num, page_content in grouped_pages.items():
            self.structured_pages.append({
                "page_num" : page_num,
                "content" : "\n\n".join(page_content)
            })

        print(f"Processing complete. {len(self.structured_pages)} pages structured and ready for chunking.")
        return self.structured_pages
        
# --- Testing the Pipeline ---
if __name__ == "__main__":
    sample_pdf_path = "Sample_book.pdf"
    
    if os.path.exists(sample_pdf_path):
        processor = DocumentProcessor(sample_pdf_path)
        pages = processor.process()
        
        if pages:
            print("\n--- Snippet of Page 1 ---")
            print(pages[0]["content"][:500] + "...\n---------------------------")
    else:
        print(f"Please place a PDF named '{sample_pdf_path}' in the directory to test the extractor.")