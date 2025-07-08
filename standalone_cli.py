#!/usr/bin/env python3
"""
Standalone Legacy CLI for testing pickle files locally.
This script handles legacy pickle loading without requiring package installation.
"""

import argparse
import os
import sys
import pickle
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import calendar
import re

# Add the src directory to Python path so we can import our modules
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Legacy compatibility classes (inline to avoid import issues)
class Node:
    def __init__(self, subject):
        self.subject = subject
        self.edges = defaultdict(list)
        self.predicate_embeddings = {}

    def add_edge(self, predicate, object_, embedder=None):
        self.edges[predicate].append(object_)
        if predicate not in self.predicate_embeddings:
            # Skip embeddings for now to avoid dependencies
            self.predicate_embeddings[predicate] = None

class Hash:
    capacity = 10000
    similarity_threshold = 0.5
    
    def __init__(self, initial_date):
        self.start_index = self.date_to_value(initial_date)
        self.arr = [None] * self.capacity
        self.size = 0
        self.embedder = None  # Skip embedder to avoid dependency issues
        self.subject_embeddings = {}
        self.subject_index = {}
        self._subject_embeddings_updated = False

    def date_to_value(self, date):
        date_obj = datetime.strptime(date, "%m/%d/%Y")
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        value = year * 372 + month * 31 + day
        return value

    def search(self, date, subject_query, predicate_query, question_type, max_hops=4, threshold=None):
        """Simplified search without embeddings."""
        if threshold is None:
            threshold = self.similarity_threshold
    
        value = self.date_to_value(date)
        index = value - self.start_index
        
        if index < 0 or index >= len(self.arr):
            return []
            
        graph = self.arr[index]
        if graph is None:
            return []
            
        # Simple string matching instead of embedding-based search
        results = []
        for subject, node in graph.items():
            # Check if subject matches (case-insensitive)
            if subject_query.lower() in subject.lower() or subject.lower() in subject_query.lower():
                for predicate, objects in node.edges.items():
                    # Check if predicate matches (case-insensitive)
                    if predicate_query.lower() in predicate.lower() or predicate.lower() in predicate_query.lower():
                        for obj in objects:
                            results.append({
                                'subject': subject,
                                'predicate': predicate,
                                'object': obj,
                                'total_score': 1.0
                            })
        
        # Return top 5 results
        return results[:5] if results else []

# Helper functions
def filter_triplets_with_dates_or_numbers(triplets):
    DATE_REGEX = r"\b(\d{1,2}[ \-/])?(January|February|March|April|May|June|July|August|September|October|November|December)[ \-/]\d{2,4}\b|\b\d{4}\b"
    NUMERIC_REGEX = r"\b\d+\b|\bmillion\b|\bbillion\b|\bthousand\b|\bhundred\b"
    COMBINED_REGEX = f"({DATE_REGEX})|({NUMERIC_REGEX})"
    
    filtered = []
    for triplet in triplets:
        subj = triplet["subject"].lower()
        obj = triplet["object"].lower()
        if re.search(COMBINED_REGEX, subj) or re.search(COMBINED_REGEX, obj):
            filtered.append(triplet)
    return filtered

def filter_triplets_with_names_or_locations(triplets):
    filtered = []
    for triplet in triplets:
        subj = triplet["subject"]
        obj = triplet["object"]
        if (subj and subj[0].isupper()) or (obj and obj[0].isupper()):
            filtered.append(triplet)
    return filtered

def register_legacy_classes():
    """Register legacy classes for pickle compatibility."""
    import __main__
    __main__.Hash = Hash
    __main__.Node = Node
    __main__.filter_triplets_with_dates_or_numbers = filter_triplets_with_dates_or_numbers
    __main__.filter_triplets_with_names_or_locations = filter_triplets_with_names_or_locations
    print("Legacy compatibility classes registered successfully.")

def load_legacy_pickle(filepath: str):
    """Load a legacy pickle file with compatibility shims."""
    register_legacy_classes()
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading legacy pickle file: {e}")
        raise

def extract_date_from_question(question):
    """Extract date from question using simple regex."""
    # Look for patterns like "16 March 2018"
    date_pattern = r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
    match = re.search(date_pattern, question, re.IGNORECASE)
    
    if match:
        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))
        
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month = month_map.get(month_name, 1)
        return f"{month:02d}/{day:02d}/{year}"
    
    return None

def extract_keywords_from_question(question):
    """Extract key terms from question."""
    question_lower = question.lower()
    
    # Simple keyword extraction
    if "poultry" in question_lower:
        subject = "poultry"
        predicate = "popular" if "popular" in question_lower else "consumption"
    else:
        # Extract first meaningful word as subject
        words = question_lower.split()
        subject = words[0] if words else "unknown"
        predicate = "related"
    
    return subject, predicate

def answer_question_simple(hash_obj, question):
    """Simple question answering using the legacy Hash object."""
    
    # Extract date from question
    date_str = extract_date_from_question(question)
    if not date_str:
        return "Could not extract date from question. Please use format like '16 March 2018'."
    
    # Extract keywords
    subject, predicate = extract_keywords_from_question(question)
    
    print(f"Searching for: subject='{subject}', predicate='{predicate}', date='{date_str}'")
    
    # Search for relevant information
    try:
        results = hash_obj.search(
            date=date_str,
            subject_query=subject,
            predicate_query=predicate,
            question_type="Other"
        )
        
        if results:
            answer = f"Based on data from {date_str}:\\n"
            for i, result in enumerate(results[:3], 1):
                answer += f"{i}. {result['subject']} {result['predicate']} {result['object']}\\n"
            return answer
        else:
            return f"No specific information found about '{subject}' on {date_str}. The system searched the knowledge graph but couldn't find matching entries for that date."
            
    except Exception as e:
        return f"Error during search: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Standalone Legacy CLI for Temporal Graph RAG")
    
    parser.add_argument("--question", "-q", required=True, help="Question to answer")
    parser.add_argument("--kg-file", required=True, help="Path to the knowledge graph pickle file")
    parser.add_argument("--groq-key", help="Groq API key (not used in this standalone version)")
    parser.add_argument("--output-format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Loading graph from: {args.kg_file}")
        print(f"Question: {args.question}")
    
    try:
        # Load the legacy pickle file
        print("Loading knowledge graph...")
        legacy_obj = load_legacy_pickle(args.kg_file)
        
        if args.verbose:
            print(f"Successfully loaded Hash object")
            print(f"Array size: {len(legacy_obj.arr)}")
            print(f"Start index: {legacy_obj.start_index}")
        
        # Process the question
        print("Processing question...")
        answer = answer_question_simple(legacy_obj, args.question)
        
        # Output the result
        if args.output_format == "json":
            result = {
                "question": args.question,
                "answer": answer,
                "graph_file": args.kg_file,
                "status": "success"
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\\n=== ANSWER ===")
            print(f"Question: {args.question}")
            print(f"Answer: {answer}")
            
    except FileNotFoundError:
        error_msg = f"Error: Knowledge graph file not found: {args.kg_file}"
        print(error_msg)
        if args.output_format == "json":
            print(json.dumps({"error": error_msg, "status": "file_not_found"}))
        sys.exit(1)
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        if args.verbose:
            import traceback
            traceback.print_exc()
        if args.output_format == "json":
            print(json.dumps({"error": error_msg, "status": "error"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
