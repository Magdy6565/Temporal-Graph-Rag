"""
Example usage of the Temporal Graph RAG system.
"""

import os
from temporal_graph_rag import TemporalGraphRAG

# Example 1: Basic usage
def basic_example():
    """Basic example of building and querying a temporal graph."""
    print("=== Basic Example ===")
    
    # Initialize the graph
    graph = TemporalGraphRAG(start_date="01/01/2020")
    
    # Add some sample triplets
    triplets_2020 = [
        {"subject": "Obama", "predicate": "visited", "object": "Chicago"},
        {"subject": "Tesla", "predicate": "announced", "object": "new model"},
        {"subject": "COVID-19", "predicate": "declared", "object": "pandemic"}
    ]
    
    triplets_2021 = [
        {"subject": "Biden", "predicate": "became", "object": "President"},
        {"subject": "SpaceX", "predicate": "launched", "object": "Starship"},
    ]
    
    # Insert triplets for different dates
    graph.insert_news("03/15/2020", triplets_2020)
    graph.insert_news("01/20/2021", triplets_2021)
    
    print(f"Graph size: {graph.size}")
    print(f"Unique subjects: {len(graph.subject_embeddings)}")
    
    # Search for information
    results = graph.search("03/15/2020", "Obama", "visited", "Other")
    if results:
        print(f"Found: {results[0]['subject']} {results[0]['predicate']} {results[0]['object']}")
    
    # Save the graph
    graph.save("example_graph.pkl")
    print("Graph saved to example_graph.pkl")


# Example 2: Loading from CSV
def csv_example():
    """Example of loading triplets from CSV data."""
    print("\n=== CSV Example ===")
    
    import pandas as pd
    
    # Create sample CSV data
    data = {
        'Sequential_Date': ['03/15/2020', '01/20/2021'],
        'triplets': [
            '[{"subject": "Obama", "predicate": "visited", "object": "Chicago"}]',
            '[{"subject": "Biden", "predicate": "became", "object": "President"}]'
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("sample_triplets.csv", index=False)
    print("Sample CSV created: sample_triplets.csv")
    
    # Load and process
    graph = TemporalGraphRAG(start_date="01/01/2020")
    
    for _, row in df.iterrows():
        import ast
        date = row['Sequential_Date']
        triplets = ast.literal_eval(row['triplets'])
        graph.insert_news(date, triplets)
    
    print(f"Loaded {graph.size} temporal entries")


# Example 3: Question answering (requires API keys)
def qa_example():
    """Example of question answering (requires API setup)."""
    print("\n=== Question Answering Example ===")
    
    # This requires proper API setup
    groq_api_key = os.getenv('GROQ_API_KEY')
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not all([groq_api_key, neo4j_uri, neo4j_user, neo4j_password]):
        print("API credentials not configured. Set environment variables:")
        print("- GROQ_API_KEY")
        print("- NEO4J_URI")
        print("- NEO4J_USERNAME")
        print("- NEO4J_PASSWORD")
        return
    
    # Load existing graph
    try:
        graph = TemporalGraphRAG.load("example_graph.pkl")
        
        # Set up connections
        graph.set_neo4j_client(neo4j_uri, neo4j_user, neo4j_password)
        
        # Answer questions
        questions = [
            "What did Obama do on March 15, 2020?",
            "Who became President on January 20, 2021?",
        ]
        
        for question in questions:
            try:
                answer = graph.answer_question(question)
                print(f"Q: {question}")
                print(f"A: {answer}")
                print()
            except Exception as e:
                print(f"Error answering '{question}': {e}")
                
    except FileNotFoundError:
        print("example_graph.pkl not found. Run basic_example() first.")


if __name__ == "__main__":
    # Run examples
    basic_example()
    csv_example()
    qa_example()
    
    print("\n=== Cleanup ===")
    # Clean up example files
    import os
    for file in ["example_graph.pkl", "sample_triplets.csv"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")
