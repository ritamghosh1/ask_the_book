import os
from dotenv import load_dotenv # type: ignore
from groq import Groq # type: ignore

# Load environment variables 
load_dotenv()

class RAGGenerator:
    def __init__(self):
        print("Initializing Groq LLM Client...")
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"

    def build_prompt(self, query: str, nodes: list) -> list:
        """
        This locks the AI into ONLY using the provided chunks.
        """
        system_prompt = (
            "You are an expert academic assistant. Your task is to provide a highly detailed, "
            "comprehensive, and well-structured answer to the user's question "
            "using ONLY the information provided in the <context> blocks below. "
            "Explain the concepts thoroughly, using bullet points or steps where appropriate to ensure clarity. "
            "If the answer cannot be deduced from the context, you must reply exactly with: "
            "'No context found from the book' Do not use outside knowledge or hallucinate.\n\n"
            "At the very end of your answer, you MUST include a citation "
            "listing the source pages used, formatted exactly like: 'Sources: pages X, Y'."
        )

        # Build the context string by injecting the text and page numbers
        context_str = ""
        for node in nodes:
            page = node.metadata.get("page", "Unknown")
            context_str += f"<context source_page='{page}'>\n{node.text}\n</context>\n\n"

        # Combine them for the LLM
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def generate(self, query: str, nodes: list):
        print("Sending context and query to Llama 3.1...")
        messages = self.build_prompt(query, nodes)
        
        # We use a low temperature (0.2) to make the AI more factual and less "creative"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2, 
        )
        
        return response.choices[0].message.content

    def generate_stream(self, query: str, nodes: list):
        """Streaming LLM Generator"""
        messages = self.build_prompt(query, nodes)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2, 
            stream=True # Tells groq to stream the response
        )
        
        # Yield tokens one by one as they arrive from Groq
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# --- Testing the Full Pipeline ---
if __name__ == "__main__":
    from embed import VectorDBManager
    from retriever import HybridRetriever
    
    sample_pdf_path = "Sample_book.pdf"
    
    if os.path.exists(sample_pdf_path):
        # 1. Ingest & Embed (RAM)
        manager = VectorDBManager()
        index, nodes = manager.process_and_store(sample_pdf_path) # type: ignore
        
        if index and nodes:
            # 2. Retrieve
            retriever = HybridRetriever(index, nodes)
            question = "What is the best way to sort an array"
            
            # The retriever returns a list of tuples: [(node, score), (node, score)...]
            best_chunk_tuples = retriever.retrieve(question, top_k=5)
            
            # We just need the raw nodes for the LLM
            just_nodes = [node for node, score in best_chunk_tuples]
            
            # 3. Generate Answer
            generator = RAGGenerator()
            final_answer = generator.generate(question, just_nodes)
            
            print("\n --- FINAL LLM ANSWER --- \n")
            print(final_answer)
            print("\n------------------------------\n")
    else:
        print(f"Please place a PDF named '{sample_pdf_path}' in the directory.")