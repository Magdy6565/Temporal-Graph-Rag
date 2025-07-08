"""
Core temporal graph implementation for time-aware knowledge storage and retrieval.
"""

import pickle
import calendar
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer, util
from .neo4j_client import Neo4jClient
from ..utils.date_utils import normalize_date
from ..utils.text_utils import filter_triplets_with_dates_or_numbers

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class Node:
    """Represents a node in the temporal knowledge graph."""
    
    def __init__(self, subject: str):
        """
        Initialize a node with a subject.
        
        Args:
            subject: The subject entity this node represents
        """
        self.subject = subject
        self.edges = defaultdict(list)  # predicate -> list of objects
        self.predicate_embeddings = {}  # predicate -> embedding vector

    def add_edge(self, predicate: str, object_: str, embedder: SentenceTransformer):
        """
        Add an edge (relation) from this node to an object.
        
        Args:
            predicate: The relation/predicate
            object_: The target object
            embedder: Sentence transformer for generating embeddings
        """
        self.edges[predicate].append(object_)
        if predicate not in self.predicate_embeddings:
            self.predicate_embeddings[predicate] = embedder.encode(
                predicate, convert_to_numpy=True
            )


class TemporalGraphRAG:
    """
    Main class for Temporal Graph RAG system.
    
    Implements a hash-based temporal knowledge graph with semantic similarity search
    and integration with Neo4j for broader contextual queries.
    """
    
    def __init__(self, start_date: str, capacity: int = 10000, 
                 similarity_threshold: float = 0.5, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Temporal Graph RAG system.
        
        Args:
            start_date: Starting date in MM/DD/YYYY format
            capacity: Maximum capacity of the temporal array
            similarity_threshold: Threshold for semantic similarity matching
            model_name: Name of the sentence transformer model
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.start_index = self._date_to_value(start_date)
        self.arr = [None] * self.capacity
        self.size = 0
        self.embedder = SentenceTransformer(model_name)
        self.subject_embeddings = {}  # subject -> embedding mapping
        self.subject_index = None     # FAISS index for fast similarity search
        self._subject_embeddings_updated = False  # Track if embeddings need reindexing
        self.neo4j_client = None

    def _date_to_value(self, date: str) -> int:
        """
        Convert date string to numerical value for hashing.
        
        Args:
            date: Date string in MM/DD/YYYY format
            
        Returns:
            Numerical value representing the date
        """
        date_obj = datetime.strptime(date, "%m/%d/%Y")
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        value = year * 372 + month * 31 + day
        return value

    def set_neo4j_client(self, uri: str, username: str, password: str):
        """
        Set up Neo4j client for semantic search.
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
        """
        self.neo4j_client = Neo4jClient(uri, username, password)

    def insert_news(self, date: str, triplets: List[Dict[str, str]]):
        """
        Insert news triplets for a specific date.
        
        Args:
            date: Date string in MM/DD/YYYY format
            triplets: List of triplet dictionaries with keys: subject, predicate, object
        """
        value = self._date_to_value(date)
        index = value - self.start_index

        if index < 0 or index >= self.capacity:
            raise ValueError(f"Date {date} is out of range for this graph")

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
                    self.subject_embeddings[subject] = self.embedder.encode(
                        subject, convert_to_numpy=True
                    )
                    self._subject_embeddings_updated = True

            graph[subject].add_edge(predicate, object_, self.embedder)

        self.size += 1

    def _build_subject_index(self):
        """Build a FAISS index for fast subject similarity search."""
        if not self._subject_embeddings_updated or not FAISS_AVAILABLE:
            if not FAISS_AVAILABLE:
                print("FAISS not available, using brute-force search")
            return
            
        embeddings = np.array(list(self.subject_embeddings.values())).astype('float32')
        self.subject_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.subject_index.add(embeddings)
        self._subject_embeddings_updated = False
        print("Subject index rebuilt")

    def find_similar_subjects(self, query_subject: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Find all subjects similar to query_subject, sorted by similarity.
        
        Args:
            query_subject: Subject to find similarities for
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            List of tuples (subject, similarity_score) sorted by similarity
        """
        if threshold is None:
            threshold = self.similarity_threshold

        if not self.subject_embeddings:
            return []

        query_embedding = self.embedder.encode(query_subject, convert_to_numpy=True)
        
        # Try using FAISS if available
        if self.subject_index is not None:
            self._build_subject_index()
            query_embedding_faiss = np.array([query_embedding]).astype('float32')
            D, I = self.subject_index.search(query_embedding_faiss, len(self.subject_embeddings))
            
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
    
    def search(self, date: str, subject_query: str, predicate_query: str, 
               question_type: str, max_hops: int = 4, threshold: Optional[float] = None) -> List[Dict]:
        """
        Search with soft matching for both subject and predicate.
        
        Args:
            date: Date to search in MM/DD/YYYY format
            subject_query: Subject to search for
            predicate_query: Predicate to search for
            question_type: Type of question (Date, Person, Other)
            max_hops: Maximum hops for multi-hop search
            threshold: Similarity threshold
            
        Returns:
            List of matching triplets with scores
        """
        if threshold is None:
            threshold = self.similarity_threshold
    
        value = self._date_to_value(date)
        index = value - self.start_index
        
        if index < 0 or index >= self.capacity or self.arr[index] is None:
            return []
            
        graph = self.arr[index]
        
        # Extract all triplets for filtering
        mytriplets = []
        for subject, node in graph.items():
            for predicate, objects in node.edges.items():
                for obj in objects:
                    mytriplets.append({
                        "subject": subject, 
                        "predicate": predicate, 
                        "object": obj
                    })
        
        # Apply question-type specific filtering
        if question_type == "Date":
            mytriplets = filter_triplets_with_dates_or_numbers(mytriplets)
        elif question_type == "Person":
            mytriplets = filter_triplets_with_dates_or_numbers(mytriplets)  # Placeholder for person filtering

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
    
        # Find all matching subjects
        similar_subjects = self.find_similar_subjects(subject_query, threshold=0)
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
    
            # Multi-hop search if enabled
            if max_hops > 0:
                visited = set()
                queue = deque()
                queue.append((node, [], 0))
                
                while queue:
                    current_node, path, hop_count = queue.popleft()
                    
                    if current_node.subject in visited or hop_count >= max_hops:
                        continue
                        
                    visited.add(current_node.subject)
                    
                    for pred, objects in current_node.edges.items():
                        pred_score = util.cos_sim(
                            query_pred_embedding, 
                            current_node.predicate_embeddings[pred]
                        ).item()
                        
                        for obj in objects:
                            full_path = path + [pred]
                            results.append({
                                'subject': matched_subject,
                                'subject_score': subject_score,
                                'predicate': pred,
                                'predicate_score': pred_score,
                                'object': obj,
                                'path': full_path,
                                'hops': hop_count + 1,
                                'total_score': subject_score + pred_score
                            })
                        
                        # Add neighbors to queue
                        for obj in objects:
                            if obj in graph:
                                queue.append((graph[obj], path + [pred], hop_count + 1))
    
        # Sort all results by total score descending
        results.sort(key=lambda x: -x['total_score'])
        
        # Return results that meet the threshold if any exist
        threshold_met_results = [
            r for r in results 
            if r['subject_score'] >= threshold and r['predicate_score'] >= threshold
        ]
        
        if threshold_met_results:
            return threshold_met_results
        elif results:  # Return top result even if below threshold
            return [results[0]]
        return []

    def search_month(self, month: int, year: int, subject: str, predicate_query: str,
                     question_type: str, max_hops: int = 4, threshold: Optional[float] = None) -> List[Dict]:
        """
        Search across a whole month with soft matching.
        
        Args:
            month: Month number (1-12)
            year: Year
            subject: Subject to search for
            predicate_query: Predicate to search for
            question_type: Type of question
            max_hops: Maximum hops for multi-hop search
            threshold: Similarity threshold
            
        Returns:
            List of all matching triplets across the month
        """
        if threshold is None:
            threshold = self.similarity_threshold

        _, num_days = calendar.monthrange(year, month)
        all_results = []

        for day in range(1, num_days + 1):
            date_str = f"{month:02d}/{day:02d}/{year}"
            daily_results = self.search(
                date_str, subject, predicate_query, question_type, max_hops, threshold
            )
            all_results.extend(daily_results)

        return all_results

    def answer_question(self, question: str) -> str:
        """
        Answer a natural language question using the temporal graph.
        
        Args:
            question: Natural language question
            
        Returns:
            Generated answer string
        """
        if self.neo4j_client is None:
            raise ValueError("Neo4j client not configured. Call set_neo4j_client() first.")
            
        from .question_processor import QuestionProcessor
        processor = QuestionProcessor(self.embedder, self.neo4j_client)
        return processor.answer_question(question, self)

    def print_graph(self, date: str):
        """Print the knowledge graph for a specific date."""
        value = self._date_to_value(date)
        index = value - self.start_index
        graph = self.arr[index]

        if not graph:
            print("No news for this date.")
            return

        for subject, node in graph.items():
            print(f"Subject: {subject}")
            for predicate, objects in node.edges.items():
                for obj in objects:
                    print(f"  --[{predicate}]--> {obj}")

    def print_graphs_10_days(self, start_date: str):
        """Print graphs for 10 days starting from start_date."""
        start_value = self._date_to_value(start_date)
        start_index = start_value - self.start_index

        print(f"Graphs for 10 days starting from {start_date}")
        print("=" * 60)

        for day_offset in range(10):
            current_index = start_index + day_offset
            if current_index < 0 or current_index >= self.capacity:
                continue

            graph = self.arr[current_index]

            base_date = datetime.strptime(start_date, "%m/%d/%Y")
            current_date = base_date + timedelta(days=day_offset)
            date_str = current_date.strftime("%m/%d/%Y")

            print(f"Date: {date_str}")
            print("-" * 60)

            if graph is None:
                print("  No news for this date.")
                print("-" * 60)
                continue

            for subject, node in graph.items():
                print(f"  Subject: {subject}")
                for predicate, objects in node.edges.items():
                    for obj in objects:
                        print(f"    - [{predicate}] -> {obj}")
            print("-" * 60)

    def save(self, filepath: str):
        """Save the temporal graph to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"TemporalGraphRAG object saved to {filepath}.")

    @staticmethod
    def load(filepath: str) -> 'TemporalGraphRAG':
        """Load a temporal graph from a pickle file."""
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"TemporalGraphRAG object loaded from {filepath}.")
        return obj
