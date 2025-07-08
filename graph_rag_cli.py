#!/usr/bin/env python3
"""
Legacy CLI wrapper for backward compatibility with Kaggle-style commands.
This script provides the same interface as your original graph_rag_cli.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from temporal_graph_rag import TemporalGraphRAG
from temporal_graph_rag.core.question_processor import QuestionProcessor
from temporal_graph_rag.legacy_compat import register_legacy_classes, load_legacy_pickle


def main():
    parser = argparse.ArgumentParser(description="Temporal Graph RAG CLI (Legacy Interface)")
    
    # Required arguments
    parser.add_argument("-q", "--question", required=True, 
                       help="Question to answer")
    parser.add_argument("--kg-file", required=True,
                       help="Path to the knowledge graph pickle file")
    parser.add_argument("--groq-key", required=True,
                       help="Groq API key")
    
    # Optional arguments
    parser.add_argument("--neo4j-uri", 
                       help="Neo4j database URI")
    parser.add_argument("--neo4j-user",
                       help="Neo4j username") 
    parser.add_argument("--neo4j-password",
                       help="Neo4j password")
    parser.add_argument("--output-format", choices=["text", "json"], default="text",
                       help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Loading graph from: {args.kg_file}")
        print(f"Question: {args.question}")
    
    try:
        # Load the temporal graph
        print("Loading knowledge graph...")
        
        # First try to load with legacy compatibility
        try:
            graph = TemporalGraphRAG.load(args.kg_file)
        except (AttributeError, ModuleNotFoundError) as e:
            if "Hash" in str(e):
                print("Detected legacy pickle format. Attempting compatibility loading...")
                register_legacy_classes()
                legacy_obj = load_legacy_pickle(args.kg_file)
                
                # Convert to new format temporarily
                temp_file = args.kg_file + ".converted.pkl"
                print(f"Converting legacy format to new format: {temp_file}")
                
                from temporal_graph_rag.legacy_compat import convert_legacy_to_new_format
                convert_legacy_to_new_format(legacy_obj, temp_file)
                
                # Load the converted file
                graph = TemporalGraphRAG.load(temp_file)
                print("Legacy file converted and loaded successfully!")
            else:
                raise e
        
        if args.verbose:
            print(f"Graph loaded successfully")
            print(f"Graph size: {graph.size}")
            print(f"Unique subjects: {len(graph.subject_embeddings)}")
        
        # Set up Neo4j if credentials provided
        if args.neo4j_uri and args.neo4j_user and args.neo4j_password:
            graph.set_neo4j_client(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
            if args.verbose:
                print("Neo4j client configured")
        elif args.verbose:
            print("Neo4j not configured - some queries may have limited functionality")
        
        # Initialize question processor
        processor = QuestionProcessor(
            embedder=graph.embedder,
            neo4j_client=graph.neo4j_client,
            groq_api_key=args.groq_key
        )
        
        # Process the question
        print("Processing question...")
        answer = processor.answer_question(args.question, graph)
        
        # Output the result
        if args.output_format == "json":
            import json
            result = {
                "question": args.question,
                "answer": answer,
                "graph_file": args.kg_file
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\nQuestion: {args.question}")
            print(f"Answer: {answer}")
            
    except FileNotFoundError:
        print(f"Error: Knowledge graph file not found: {args.kg_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
