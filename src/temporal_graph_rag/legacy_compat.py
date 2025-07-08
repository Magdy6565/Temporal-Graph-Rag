"""
Legacy compatibility module for loading old pickle files.

This module provides compatibility shims to load pickle files created
with the original notebook code structure, exactly matching the notebook classes.
"""

import pickle
import sys
import re
import calendar
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sentence_transformers import SentenceTransformer, util


class Node:
    """
    Legacy Node class compatibility shim.
    
    This recreates the original Node class from the notebook exactly.
    """
    
    def __init__(self, subject):
        self.subject = subject
        self.edges = defaultdict(list)  # predicate -> list of objects
        self.predicate_embeddings = {}  # predicate -> embedding vector

    def add_edge(self, predicate, object_, embedder):
        self.edges[predicate].append(object_)
        if predicate not in self.predicate_embeddings:
            self.predicate_embeddings[predicate] = embedder.encode(predicate, convert_to_numpy=True)


# Additional helper functions from the notebook
def filter_triplets_with_dates_or_numbers(triplets):
    """Filter triplets that contain dates or numbers."""
    # Basic regex for detecting dates, numbers, and number words
    DATE_REGEX = r"\b(\d{1,2}[ \-/])?(January|February|March|April|May|June|July|August|September|October|November|December)[ \-/]\d{2,4}\b|\b\d{4}\b|\b\d{1,4}[ ]?B\.?C\.?\b"
    NUMERIC_REGEX = r"\b\d+\b|\bmillion\b|\bbillion\b|\bthousand\b|\bhundred\b|\bfew\b|\bseveral\b|\bone\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b|\bsix\b|\bseven\b|\beight\b|\bnine\b|\bten\b"
    
    # Combine both into one for checking subject or object
    COMBINED_REGEX = f"({DATE_REGEX})|({NUMERIC_REGEX})"
    
    filtered = []
    for triplet in triplets:
        subj = triplet["subject"].lower()
        obj = triplet["object"].lower()
        if re.search(COMBINED_REGEX, subj) or re.search(COMBINED_REGEX, obj):
            filtered.append(triplet)
    return filtered


def filter_triplets_with_names_or_locations(triplets):
    """Filter triplets that contain names or locations (simplified version)."""
    # This is a simplified version - in practice you'd use NER
    filtered = []
    for triplet in triplets:
        subj = triplet["subject"]
        obj = triplet["object"]
        # Simple heuristic: if it starts with capital letter, might be a name/location
        if (subj and subj[0].isupper()) or (obj and obj[0].isupper()):
            filtered.append(triplet)
    return filtered


class Hash:
    """
    Legacy Hash class compatibility shim.
    
    This recreates the original Hash class from the notebook exactly.
    """
    
    capacity = 10000
    similarity_threshold = 0.5
    
    def __init__(self, initial_date):
        self.start_index = self.date_to_value(initial_date)
        self.arr = [None] * self.capacity
        self.size = 0
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.subject_embeddings = {}  # New: subject -> embedding mapping
        self.subject_index = {}       # New: for fast similarity search
        self._subject_embeddings_updated = False  # Track if embeddings need reindexing
        

    def date_to_value(self, date):
        date_obj = datetime.strptime(date, "%m/%d/%Y")
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        value = year * 372 + month * 31 + day
        return value

    def insert_news(self, date, triplets):
        value = self.date_to_value(date)
        index = value - self.start_index

        if self.arr[index] is None:
            self.arr[index] = {}  # New graph: subject -> Node

        graph = self.arr[index]

        for triplet in triplets:
            subject = triplet['subject']
            predicate = triplet['predicate']
            object_ = triplet['object']

            if subject not in graph:
                graph[subject] = Node(subject)
                # Add subject to embedding store if not already present
                if subject not in self.subject_embeddings:
                    self.subject_embeddings[subject] = self.embedder.encode(subject, convert_to_numpy=True)
                    self._subject_embeddings_updated = True

            graph[subject].add_edge(predicate, object_, self.embedder)

        self.size += 1

    def _build_subject_index(self):
        """Build a FAISS index for fast subject similarity search"""
        if not self._subject_embeddings_updated:
            return
            
        try:
            import faiss
        except ImportError:
            print("FAISS not available, using brute-force search")
            self.subject_index = None
            self._subject_embeddings_updated = False
            return

        embeddings = np.array(list(self.subject_embeddings.values())).astype('float32')
        self.subject_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.subject_index.add(embeddings)
        self._subject_embeddings_updated = False
        print("Subject index rebuilt")

    def find_similar_subjects(self, query_subject, threshold=None):
        """Find all subjects similar to query_subject, sorted by similarity"""
        if threshold is None:
            threshold = self.similarity_threshold

        if not self.subject_embeddings:
            return []

        query_embedding = self.embedder.encode(query_subject, convert_to_numpy=True)
        
        # Try using FAISS if available
        if self.subject_index is not None:
            self._build_subject_index()
            query_embedding = np.array([query_embedding]).astype('float32')
            D, I = self.subject_index.search(query_embedding, len(self.subject_embeddings))
            
            subjects = list(self.subject_embeddings.keys())
            results = []
            for i, score in zip(I[0], D[0]):
                if i == -1 or score < threshold:
                    continue
                results.append((subjects[i], score))
            return sorted(results, key=lambda x: -x[1])
        else:
            # Fallback to brute-force search
            results = []
            for subject, embedding in self.subject_embeddings.items():
                score = util.cos_sim(query_embedding, embedding).item()
                if score >= threshold:
                    results.append((subject, score))
            return sorted(results, key=lambda x: -x[1])
    
    def search(self, date, subject_query, predicate_query, question_type, max_hops=4, threshold=None):
        """Search with soft matching for both subject and predicate."""
        if threshold is None:
            threshold = self.similarity_threshold
    
        value = self.date_to_value(date)
        index = value - self.start_index
        
        if index < 0 or index >= len(self.arr):
            return []
            
        graph = self.arr[index]
        if graph is None:
            return []
            
        mytriplets = []
        for subject, node in graph.items():
            for predicate, objects in node.edges.items():
                for obj in objects:
                    mytriplets.append({"subject": subject, "predicate": predicate, "object": obj})
        
        if question_type == "Date":
            mytriplets = filter_triplets_with_dates_or_numbers(mytriplets)
        elif question_type == "Person":
            mytriplets = filter_triplets_with_names_or_locations(mytriplets)

        if question_type in ["Date", "Person"]:
            filtered_graph = {}
            for triplet in mytriplets:
                subject = triplet['subject']
                predicate = triplet['predicate']
                obj = triplet['object']
                
                if subject not in filtered_graph:
                    filtered_graph[subject] = Node(subject)
                    if subject in self.subject_embeddings:
                        filtered_graph[subject].predicate_embeddings = graph[subject].predicate_embeddings.copy()
                
                filtered_graph[subject].edges[predicate].append(obj)
                if predicate not in filtered_graph[subject].predicate_embeddings:
                    filtered_graph[subject].predicate_embeddings[predicate] = \
                        graph[subject].predicate_embeddings.get(predicate, 
                        self.embedder.encode(predicate, convert_to_numpy=True))
        
            graph = filtered_graph
    
        # Find all matching subjects (including exact match)
        similar_subjects = self.find_similar_subjects(subject_query, threshold=0)  # Get all possible matches
        if not similar_subjects:
            return []
    
        results = []
        query_pred_embedding = self.embedder.encode(predicate_query, convert_to_numpy=True)
        
        for matched_subject, subject_score in similar_subjects:
            if matched_subject not in graph:
                continue
    
            node = graph[matched_subject]
            
            # Check for exact predicate match first
            if predicate_query in node.edges:
                for obj in node.edges[predicate_query]:
                    results.append({
                        'subject': matched_subject,
                        'subject_score': subject_score,
                        'predicate': predicate_query,
                        'predicate_score': 1.0,
                        'object': obj,
                        'path': None,
                        'hops': 0,
                        'total_score': subject_score + 1.0
                    })
            
            # Perform soft matching for predicates
            for pred, emb in node.predicate_embeddings.items():
                pred_score = util.cos_sim(query_pred_embedding, emb).item()
                # Only proceed if there are objects for this predicate
                if pred in node.edges:
                    for obj in node.edges[pred]:
                        results.append({
                            'subject': matched_subject,
                            'subject_score': subject_score,
                            'predicate': pred,
                            'predicate_score': pred_score,
                            'object': obj,
                            'path': None,
                            'hops': 0,
                            'total_score': subject_score + pred_score
                        })
    
        # Sort all results by total score descending
        results.sort(key=lambda x: -x['total_score'])
        
        # Return results that meet the threshold if any exist
        threshold_met_results = [r for r in results if r['subject_score'] >= threshold and r['predicate_score'] >= threshold]
        
        if threshold_met_results:
            return threshold_met_results
        elif results:  # Return top result even if below threshold
            return [results[0]]
        return []

    def search_month(self, month, year, subject, predicate_query, question_type, max_hops=4, threshold=None):
        """Search across a whole month with soft matching"""
        if threshold is None:
            threshold = self.similarity_threshold

        _, num_days = calendar.monthrange(year, month)
        all_results = []

        for day in range(1, num_days + 1):
            date_str = f"{month:02d}/{day:02d}/{year}"
            daily_results = self.search(date_str, subject, predicate_query, question_type, max_hops, threshold)
            all_results.extend(daily_results)

        return all_results

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Hash object saved to {filepath}.")

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"Hash object loaded from {filepath}.")
        return obj


def register_legacy_classes():
    """
    Register legacy classes for pickle compatibility.
    
    This function should be called before attempting to load
    legacy pickle files.
    """
    # Register the classes in the main module namespace
    # This allows pickle to find it when unpickling
    try:
        import temporal_graph_rag.__main__
        temporal_graph_rag.__main__.Hash = Hash
        temporal_graph_rag.__main__.Node = Node
        temporal_graph_rag.__main__.filter_triplets_with_dates_or_numbers = filter_triplets_with_dates_or_numbers
        temporal_graph_rag.__main__.filter_triplets_with_names_or_locations = filter_triplets_with_names_or_locations
    except:
        pass
    
    # Also register in current module for safety
    try:
        import __main__
        __main__.Hash = Hash
        __main__.Node = Node
        __main__.filter_triplets_with_dates_or_numbers = filter_triplets_with_dates_or_numbers
        __main__.filter_triplets_with_names_or_locations = filter_triplets_with_names_or_locations
    except:
        pass
    
    # Register in sys.modules for broader compatibility
    sys.modules['__main__'].Hash = Hash
    sys.modules['__main__'].Node = Node
    sys.modules['__main__'].filter_triplets_with_dates_or_numbers = filter_triplets_with_dates_or_numbers
    sys.modules['__main__'].filter_triplets_with_names_or_locations = filter_triplets_with_names_or_locations
    
    print("Legacy compatibility classes registered successfully.")


def load_legacy_pickle(filepath: str) -> Any:
    """
    Load a legacy pickle file with compatibility shims.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        The unpickled object
        
    Raises:
        Exception: If loading fails even with compatibility shims
    """
    # Register legacy classes first
    register_legacy_classes()
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading legacy pickle file: {e}")
        print("This might be due to incompatible pickle formats.")
        print("Consider re-training the model with the new package structure.")
        raise


def convert_legacy_to_new_format(legacy_hash: Hash, output_path: str):
    """
    Convert a legacy Hash object to the new TemporalGraphRAG format.
    
    Args:
        legacy_hash: The legacy Hash object
        output_path: Path to save the new format
    """
    from temporal_graph_rag.core.temporal_graph import TemporalGraphRAG
    
    # Create new TemporalGraphRAG instance
    start_date = "01/01/2000"  # Default start date
    new_rag = TemporalGraphRAG(start_date=start_date)
    
    # Convert the data
    for index, date_graph in enumerate(legacy_hash.arr):
        if date_graph is None:
            continue
            
        # Calculate the actual date from the index
        date_value = legacy_hash.start_index + index
        
        # Convert back to date format
        # This is the reverse of the date_to_value function
        year = date_value // 372
        remainder = date_value % 372
        month = remainder // 31
        day = remainder % 31
        
        # Handle edge cases
        if month == 0:
            month = 1
        if day == 0:
            day = 1
            
        date_str = f"{month:02d}/{day:02d}/{year}"
        
        # Convert to triplet format
        triplets = []
        for subject, node in date_graph.items():
            for predicate, obj_list in node.edges.items():
                for obj in obj_list:
                    triplets.append({
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj
                    })
        
        # Insert into new format
        if triplets:
            new_rag.insert_news(date_str, triplets)
    
    # Save in new format
    new_rag.save(output_path)
    print(f"Successfully converted legacy format to new format: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Legacy compatibility utilities")
    parser.add_argument("--convert", help="Convert legacy pickle to new format")
    parser.add_argument("--input", help="Input legacy pickle file")
    parser.add_argument("--output", help="Output new format file")
    
    args = parser.parse_args()
    
    if args.convert and args.input and args.output:
        try:
            legacy_obj = load_legacy_pickle(args.input)
            convert_legacy_to_new_format(legacy_obj, args.output)
        except Exception as e:
            print(f"Conversion failed: {e}")
    else:
        print("Usage: python legacy_compat.py --convert --input legacy.pkl --output new.pkl")
