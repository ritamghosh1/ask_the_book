import os
import time
from embed import VectorDBManager
from retriever import HybridRetriever
from llm import RAGGenerator

def run_e2e_latency_test(retriever, generator, test_queries: list):
    print("="*60)
    print("⏳ Phase 2: Retrieval & Generation Latency\n")

    for i, query in enumerate(test_queries, 1):
        print(f"---  Test {i}: '{query}' ---")
        
        retrieval_start = time.time()
        best_chunk_tuples = retriever.retrieve(query, top_k=3)
        just_nodes = [node for node, score in best_chunk_tuples]
        retrieval_time = time.time() - retrieval_start
        
        llm_start = time.time()
        answer = generator.generate(query, just_nodes)
        llm_time = time.time() - llm_start
        
        print(f"Successful End-to-End Run")
        print(f"Retrieval & Rerank Time: {retrieval_time:.3f} seconds")
        print(f"LLM Generation Time:   {llm_time:.3f} seconds")
        print(f"Total Query Latency:   {retrieval_time + llm_time:.3f} seconds")
        print(f"Answer Snippet: {answer[:150]}...")
        print("-" * 60 + "\n")

def run_tiered_evaluation(retriever, generator, qa_pairs, default_top_k=5):
    print("\nRunning Tiered Evaluation...\n")
    
    results_by_difficulty = {"easy": [], "medium": [], "hard": [], "negative": []}
    SLEEP_SECONDS = 5
    
    for i, pair in enumerate(qa_pairs):
        query = pair["query"]
        expected = pair["expected_substring"]
        difficulty = pair["difficulty"]
        top_k = pair.get("top_k_override", default_top_k)  # ← use override if present
        
        best_chunks = retriever.retrieve(query, top_k=top_k)
        just_nodes = [node for node, score in best_chunks]
        
        answer = generator.generate(query, just_nodes)
        
        hit = expected.lower() in answer.lower()
        results_by_difficulty[difficulty].append(hit)
        
        status = "✅" if hit else "❌"
        print(f"Q{i+1} [{difficulty.upper():8}] {status} | {query[:55]}...")
        if not hit:
            print(f"           Expected '{expected}' in answer")
            print(f"           Got: {answer[:120]}...")
        
        print(f"    Waiting {SLEEP_SECONDS}s to respect Groq rate limit...")
        time.sleep(SLEEP_SECONDS)
    
    print("\n" + "="*60)
    print(" RESULTS BY DIFFICULTY TIER")
    print("="*60)
    
    total_pass = 0
    total_all = 0
    for tier in ["easy", "medium", "hard", "negative"]:
        results = results_by_difficulty[tier]
        if not results:
            continue
        passed = sum(results)
        total = len(results)
        rate = passed / total * 100
        bar = "█" * passed + "░" * (total - passed)
        print(f"{tier.upper():10} | {bar} | {passed}/{total} ({rate:.0f}%)")
        total_pass += passed
        total_all += total
    
    print("-"*60)
    print(f"{'OVERALL':10} | {total_pass}/{total_all} ({total_pass/total_all*100:.0f}%)")

if __name__ == "__main__":
    test_pdf = "Sample_book.pdf"
    
    latency_questions = [
        "What is Insertion Sort?",
        "How to handle overflows in hashing?",
        "What is the capital of France?" 
    ]
    
    evaluation_qa_pairs = [
    {
        "query": "What is Insertion Sort?",
        "expected_substring": "smallest element",
        "difficulty": "easy",
    },
    {
        "query": "What is a collision in hashing?",
        "expected_substring": "same cell or bucket",  
        "difficulty": "easy",
    },
    {
        "query": "When two keys produce the same hash output, what are they called?",
        "expected_substring": "synonyms",
        "difficulty": "medium",
    },
    {
        "query": "Which sorting method works by repeatedly finding the minimum and swapping it into position?",
        "expected_substring": "Selection Sort",
        "difficulty": "medium",
    },
    {
        "query": "What is the worst case and best case time complexity of Quick Sort, and what causes each?",
        "expected_substring": "O(N^2)",
        "difficulty": "hard",
    },
    {
        "query": "How does the pivot selection strategy affect Quick Sort performance?",
        "expected_substring": "Median-of-Three",
        "difficulty": "hard",
        "top_k_override": 5,  
    },
    {
        "query": "What are the two ways to handle overflow in a hash table and how do they differ?",
        "expected_substring": "Open Addressing",
        "difficulty": "hard",
    },
    {
        "query": "What is the time complexity of Dijkstra's algorithm?",
        "expected_substring": "No context found",
        "difficulty": "negative",
    },
    {
        "query": "Explain how a neural network learns using backpropagation.",
        "expected_substring": "No context found",
        "difficulty": "negative",
    },
    {
        "query": "Who invented the Quick Sort algorithm and in what year?",
        "expected_substring": "No context found",  # ← fixed, LLM actually says this
        "difficulty": "negative",
    },
]

    # --- Build shared components once ---
    manager = VectorDBManager()
    index, nodes = manager.process_and_store(test_pdf)  # type: ignore
    retriever = HybridRetriever(index, nodes)
    generator = RAGGenerator()

    run_e2e_latency_test(retriever, generator, latency_questions)
    run_tiered_evaluation(retriever, generator, evaluation_qa_pairs)