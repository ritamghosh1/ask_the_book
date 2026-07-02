import os
from rank_bm25 import BM25Okapi # type: ignore
from sentence_transformers import CrossEncoder # type: ignore

class HybridRetriever:
    def __init__(self, index, nodes):
        # 1. Initialize the Semantic Vector Retriever (Grabs top 15 by meaning)
        print("Initializing Vector Retriever...")
        self.vector_retriever = index.as_retriever(similarity_top_k=15)
        
        # 2. Initialize the BM25 Keyword Retriever (Grabs top 15 by exact words)
        print("Initializing BM25 Keyword Retriever...")
        self.nodes = nodes
        # Tokenize the text (split into lowercase words) for the BM25 index
        tokenized_corpus = [node.text.lower().split() for node in self.nodes]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 3. Initialize the Cross-Encoder Reranker
        print("Initializing Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

    def rrf_fusion(self, vector_results, bm25_results, k=60):
        """
        Combines two ranked lists. A chunk gets a higher score if it appears high on BOTH lists.
        Formula: score = 1 / (k + rank)
        """
        fused_scores = {}
        node_map = {} # Helper to keep track of the actual node objects

        # Score the Vector results
        for rank, node_with_score in enumerate(vector_results):
            node = node_with_score.node
            node_id = node.node_id
            node_map[node_id] = node
            
            if node_id not in fused_scores:
                fused_scores[node_id] = 0.0
            fused_scores[node_id] += 1.0 / (k + rank + 1) # rank is 0-indexed

        # Score the BM25 results
        for rank, node in enumerate(bm25_results):
            node_id = node.node_id
            node_map[node_id] = node
            
            if node_id not in fused_scores:
                fused_scores[node_id] = 0.0
            fused_scores[node_id] += 1.0 / (k + rank + 1)

        # Sort the dictionary by the fused scores in descending order
        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return a deduplicated list of just the node objects
        return [node_map[node_id] for node_id, score in reranked_results]

    def retrieve(self, query: str, top_k=3):
        print(f"\nQuery: '{query}'")
        
        # Step 1: Get Semantic matches
        vector_results = self.vector_retriever.retrieve(query)
        
        # Step 2: Get Keyword matches
        tokenized_query = query.lower().split()
        bm25_results = self.bm25.get_top_n(tokenized_query, self.nodes, n=15)
        
        # Step 3: Fuse them together!
        fused_nodes = self.rrf_fusion(vector_results, bm25_results)
        print(f"Fused and deduplicated down to {len(fused_nodes)} candidate chunks.")

        # Step 4: Cross-Encoder Reranking
        print("Reranking with Cross-Encoder...")
        # Format the input for the Cross-Encoder: [[Query, Chunk 1], [Query, Chunk 2], ...]
        cross_inp = [[query, node.text] for node in fused_nodes]
        cross_scores = self.reranker.predict(cross_inp)

        # Zip the nodes and their new scores together, then sort by the highest score
        scored_nodes = list(zip(fused_nodes, cross_scores))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        print("\n--- RERANKER DEBUG LOG ---")
        for rank, (node, score) in enumerate(scored_nodes[:5]):
            print(f"Rank {rank+1} | Score: {score:.2f} | Snippet: {node.text[:60]}...")
        print("-----------------------------\n")
        
        best_nodes = scored_nodes[:top_k]
        return best_nodes

# --- Testing the Pipeline ---
if __name__ == "__main__":
    from embed import VectorDBManager
    
    sample_pdf_path = "Sample_book.pdf"
    
    if os.path.exists(sample_pdf_path):
        # 1. Run the embedding pipeline first (since it's all in RAM)
        manager = VectorDBManager()
        index, nodes = manager.process_and_store(sample_pdf_path) # type: ignore
        
        if index and nodes:
            # 2. Load the Hybrid Retriever
            retriever = HybridRetriever(index, nodes)
            
            # 3. Ask a question!
            question = "How to hit a ball for a six ?"
            results = retriever.retrieve(question, top_k=3)
            
            # 4. Print the final results
            print("\n--- TOP 3 RERANKED RESULTS ---\n")
            for i, (node, score) in enumerate(results):
                page = node.metadata.get("page", "Unknown")
                print(f"Result {i+1} (Reranker Score: {score:.4f} | Source: Page {page})")
                print(f"{node.text[:250]}...\n")
    else:
        print(f"Please place a PDF named '{sample_pdf_path}' in the directory.")