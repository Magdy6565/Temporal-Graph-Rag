"""
Legacy compatibility module for loading old pickle files.

This module provides compatibility shims to load pickle files created
with the original notebook code structure.
"""

import pickle
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime


class Hash:
    """
    Legacy Hash class compatibility shim.
    
    This recreates the original Hash class from the notebook to allow
    loading of pickle files created with the old structure.
    """
    
    def __init__(self, start_date: str = "01/01/2000"):
        """Initialize Hash with start date."""
        self.start_date = start_date
        self.date_to_number = {}
        self.number_to_date = {}
        self.graph = {}
        self.current_number = 0
        
        # Initialize with start date
        self._add_date(start_date)
    
    def _add_date(self, date_str: str) -> int:
        """Add a date to the hash mapping."""
        if date_str not in self.date_to_number:
            self.date_to_number[date_str] = self.current_number
            self.number_to_date[self.current_number] = date_str
            self.current_number += 1
        return self.date_to_number[date_str]
    
    def insert_news(self, date: str, news_list: List[Dict[str, str]]):
        """Insert news for a specific date."""
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
        """Search for information in the graph."""
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


def register_legacy_classes():
    """
    Register legacy classes for pickle compatibility.
    
    This function should be called before attempting to load
    legacy pickle files.
    """
    # Register the Hash class in the main module namespace
    # This allows pickle to find it when unpickling
    import temporal_graph_rag.__main__
    temporal_graph_rag.__main__.Hash = Hash
    
    # Also register in current module for safety
    sys.modules['__main__'].Hash = Hash
    
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
    new_rag = TemporalGraphRAG(start_date=legacy_hash.start_date)
    
    # Convert the data
    for date_num, date_graph in legacy_hash.graph.items():
        date_str = legacy_hash.number_to_date[date_num]
        
        # Convert to triplet format
        triplets = []
        for subject, pred_dict in date_graph.items():
            for predicate, obj_list in pred_dict.items():
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
