"""
Standalone Legacy Pickle Fixer for Kaggle

This script can be downloaded and run on Kaggle to fix legacy pickle loading issues.
It recreates the original Hash class and provides utilities to convert old pickle files.

Usage on Kaggle:
1. Download this script to your Kaggle notebook
2. Run the functions to convert your legacy pickle files
3. Use the converted files with the new package

Example:
    # In a Kaggle cell:
    exec(open("kaggle_pickle_fix.py").read())
    
    # Convert your legacy file
    converted_data = fix_legacy_pickle("/kaggle/input/your-dataset/model.pkl")
    
    # Save in new format
    import pickle
    with open("fixed_model.pkl", "wb") as f:
        pickle.dump(converted_data, f)
"""

import pickle
import sys
import re
import calendar
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np


class Node:
    """Recreation of the original Node class from the notebook."""
    
    def __init__(self, subject):
        self.subject = subject
        self.edges = defaultdict(list)  # predicate -> list of objects
        self.predicate_embeddings = {}  # predicate -> embedding vector

    def add_edge(self, predicate, object_, embedder=None):
        self.edges[predicate].append(object_)
        if predicate not in self.predicate_embeddings:
            # For Kaggle compatibility, we'll skip the embedder part
            self.predicate_embeddings[predicate] = None
    
    def get_all_predicates(self) -> List[str]:
        return list(self.edges.keys())
    
    def set_predicate_embedding(self, predicate: str, embedding):
        self.predicate_embeddings[predicate] = embedding
    
    def get_predicate_embedding(self, predicate: str):
        return self.predicate_embeddings.get(predicate)


# Helper functions from the notebook
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
    """Recreation of the original Hash class from the notebook."""
    
    def __init__(self, start_date: str = "01/01/2000"):
        self.start_date = start_date
        self.date_to_number = {}
        self.number_to_date = {}
        self.graph = {}
        self.current_number = 0
        self._add_date(start_date)
    
    def _add_date(self, date_str: str) -> int:
        if date_str not in self.date_to_number:
            self.date_to_number[date_str] = self.current_number
            self.number_to_date[self.current_number] = date_str
            self.current_number += 1
        return self.date_to_number[date_str]
    
    def insert_news(self, date: str, news_list: List[Dict[str, str]]):
        date_num = self._add_date(date)
        if date_num not in self.graph:
            self.graph[date_num] = {}
        
        for news in news_list:
            subject = news.get('subject', '')
            predicate = news.get('predicate', '')
            obj = news.get('object', '')
            
            if subject not in self.graph[date_num]:
                self.graph[date_num][subject] = {}
            if predicate not in self.graph[date_num][subject]:
                self.graph[date_num][subject][predicate] = []
            
            self.graph[date_num][subject][predicate].append(obj)
    
    def search(self, date: str, subject: str = None, predicate: str = None, 
               question_type: str = None) -> List[str]:
        if date not in self.date_to_number:
            return []
        
        date_num = self.date_to_number[date]
        if date_num not in self.graph:
            return []
        
        results = []
        date_graph = self.graph[date_num]
        
        if subject and subject in date_graph:
            if predicate and predicate in date_graph[subject]:
                results.extend(date_graph[subject][predicate])
            elif not predicate:
                for pred_dict in date_graph[subject].values():
                    results.extend(pred_dict)
        elif not subject:
            for subj_dict in date_graph.values():
                if predicate:
                    for pred, obj_list in subj_dict.items():
                        if pred == predicate:
                            results.extend(obj_list)
                else:
                    for pred_dict in subj_dict.values():
                        results.extend(pred_dict)
        
        return results


def register_hash_class():
    """Register the Hash and Node classes so pickle can find them."""
    import __main__
    __main__.Hash = Hash
    __main__.Node = Node
    __main__.filter_triplets_with_dates_or_numbers = filter_triplets_with_dates_or_numbers
    __main__.filter_triplets_with_names_or_locations = filter_triplets_with_names_or_locations
    print("Hash and Node classes registered successfully!")


def fix_legacy_pickle(filepath: str) -> Any:
    """
    Fix and load a legacy pickle file.
    
    Args:
        filepath: Path to the legacy pickle file
        
    Returns:
        The loaded object with Hash class properly registered
    """
    print(f"Attempting to fix legacy pickle: {filepath}")
    
    # Register the Hash class first
    register_hash_class()
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print("Successfully loaded legacy pickle file!")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Make sure the file was created with the original notebook code.")
        raise


def convert_hash_to_triplets(hash_obj: Hash) -> List[Dict]:
    """
    Convert a Hash object to a list of triplets.
    
    Args:
        hash_obj: The Hash object to convert
        
    Returns:
        List of triplet dictionaries with date information
    """
    triplets = []
    
    for date_num, date_graph in hash_obj.graph.items():
        date_str = hash_obj.number_to_date[date_num]
        
        for subject, pred_dict in date_graph.items():
            for predicate, obj_list in pred_dict.items():
                for obj in obj_list:
                    triplets.append({
                        'date': date_str,
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj
                    })
    
    print(f"Converted to {len(triplets)} triplets")
    return triplets


def save_as_simple_pickle(data: Any, output_path: str):
    """Save data as a simple pickle file without class dependencies."""
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved converted data to: {output_path}")


# Convenience function for Kaggle users
def kaggle_quick_fix(input_path: str, output_path: str = "fixed_model.pkl"):
    """
    Quick fix for Kaggle users - converts legacy pickle to simple data structure.
    
    Args:
        input_path: Path to legacy pickle file
        output_path: Path for the fixed output file
        
    Returns:
        The converted data
    """
    print("=== Kaggle Legacy Pickle Quick Fix ===")
    
    # Load the legacy file
    legacy_data = fix_legacy_pickle(input_path)
    
    if hasattr(legacy_data, 'graph'):
        # It's a Hash object, convert to triplets
        print("Detected Hash object, converting to triplets...")
        triplets = convert_hash_to_triplets(legacy_data)
        
        # Create a simple data structure
        converted_data = {
            'type': 'temporal_graph_triplets',
            'start_date': getattr(legacy_data, 'start_date', '01/01/2000'),
            'triplets': triplets,
            'date_mappings': {
                'date_to_number': getattr(legacy_data, 'date_to_number', {}),
                'number_to_date': getattr(legacy_data, 'number_to_date', {})
            }
        }
    else:
        # Unknown format, save as-is
        print("Unknown format, saving as-is...")
        converted_data = legacy_data
    
    # Save the converted data
    save_as_simple_pickle(converted_data, output_path)
    
    print(f"✅ Successfully converted {input_path} to {output_path}")
    print("You can now use this file with the new package!")
    
    return converted_data


# Example usage for Kaggle
if __name__ == "__main__":
    print("Legacy Pickle Fixer loaded!")
    print("Usage examples:")
    print("1. kaggle_quick_fix('/kaggle/input/your-dataset/model.pkl')")
    print("2. fix_legacy_pickle('/path/to/file.pkl')")
    print("3. register_hash_class() # if you want to handle it manually")
