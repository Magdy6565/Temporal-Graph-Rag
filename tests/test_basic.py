"""
Tests for the temporal graph functionality.
"""

import pytest
import tempfile
from datetime import datetime
from temporal_graph_rag import TemporalGraphRAG, Node
from temporal_graph_rag.utils.date_utils import normalize_date
from temporal_graph_rag.utils.text_utils import filter_triplets_with_dates_or_numbers


class TestTemporalGraph:
    """Test cases for TemporalGraphRAG class."""
    
    def test_node_creation(self):
        """Test Node creation and edge addition."""
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        node = Node("Obama")
        assert node.subject == "Obama"
        assert len(node.edges) == 0
        
        node.add_edge("visited", "Chicago", embedder)
        assert "visited" in node.edges
        assert "Chicago" in node.edges["visited"]
        assert "visited" in node.predicate_embeddings
    
    def test_graph_initialization(self):
        """Test graph initialization."""
        graph = TemporalGraphRAG(start_date="01/01/2020")
        assert graph.capacity == 10000
        assert graph.similarity_threshold == 0.5
        assert graph.size == 0
    
    def test_date_conversion(self):
        """Test date to value conversion."""
        graph = TemporalGraphRAG(start_date="01/01/2020")
        
        # Test known date
        value1 = graph._date_to_value("01/01/2020")
        value2 = graph._date_to_value("01/02/2020")
        
        assert value2 == value1 + 1
    
    def test_triplet_insertion(self):
        """Test inserting triplets."""
        graph = TemporalGraphRAG(start_date="01/01/2020")
        
        triplets = [
            {"subject": "Obama", "predicate": "visited", "object": "Chicago"},
            {"subject": "Biden", "predicate": "became", "object": "President"}
        ]
        
        graph.insert_news("01/15/2020", triplets)
        
        assert graph.size == 1
        assert len(graph.subject_embeddings) == 2
        assert "Obama" in graph.subject_embeddings
        assert "Biden" in graph.subject_embeddings
    
    def test_search_functionality(self):
        """Test basic search functionality."""
        graph = TemporalGraphRAG(start_date="01/01/2020")
        
        triplets = [
            {"subject": "Obama", "predicate": "visited", "object": "Chicago"}
        ]
        
        graph.insert_news("01/15/2020", triplets)
        
        # Test exact match
        results = graph.search("01/15/2020", "Obama", "visited", "Other")
        assert len(results) > 0
        assert results[0]["subject"] == "Obama"
        assert results[0]["predicate"] == "visited"
        assert results[0]["object"] == "Chicago"
    
    def test_save_and_load(self):
        """Test saving and loading graphs."""
        graph = TemporalGraphRAG(start_date="01/01/2020")
        
        triplets = [
            {"subject": "Test", "predicate": "action", "object": "result"}
        ]
        
        graph.insert_news("01/15/2020", triplets)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            graph.save(f.name)
            
            # Load the graph
            loaded_graph = TemporalGraphRAG.load(f.name)
            
            assert loaded_graph.size == graph.size
            assert len(loaded_graph.subject_embeddings) == len(graph.subject_embeddings)
            assert "Test" in loaded_graph.subject_embeddings


class TestDateUtils:
    """Test cases for date utilities."""
    
    def test_normalize_date_full(self):
        """Test full date normalization."""
        result = normalize_date("March 15, 2020")
        assert result["type"] == 0
        assert result["normalized_date"] == "03/15/2020"
    
    def test_normalize_date_month_year(self):
        """Test month/year normalization."""
        result = normalize_date("March 2020")
        assert result["type"] == 1
        assert result["normalized_date"] == "03/01/2020"
    
    def test_normalize_date_year_only(self):
        """Test year-only normalization."""
        result = normalize_date("2020")
        assert result["type"] == 3
        assert result["normalized_date"] == "01/01/2020"
    
    def test_normalize_date_invalid(self):
        """Test invalid date handling."""
        result = normalize_date("invalid date")
        assert result["type"] == 5
        assert result["normalized_date"] is None


class TestTextUtils:
    """Test cases for text utilities."""
    
    def test_filter_dates_numbers(self):
        """Test filtering triplets with dates and numbers."""
        triplets = [
            {"subject": "Obama", "predicate": "born", "object": "1961"},
            {"subject": "Event", "predicate": "happened", "object": "March 2020"},
            {"subject": "Person", "predicate": "lives", "object": "Chicago"},
            {"subject": "Company", "predicate": "has", "object": "1000 employees"}
        ]
        
        filtered = filter_triplets_with_dates_or_numbers(triplets)
        
        # Should include triplets with years, dates, and numbers
        assert len(filtered) >= 3
        subjects_objects = [t["subject"] + " " + t["object"] for t in filtered]
        assert any("1961" in so for so in subjects_objects)
        assert any("March 2020" in so for so in subjects_objects)
        assert any("1000" in so for so in subjects_objects)


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    test_graph = TestTemporalGraph()
    test_date = TestDateUtils()
    test_text = TestTextUtils()
    
    print("Running tests...")
    
    try:
        test_graph.test_node_creation()
        print("✓ Node creation test passed")
    except Exception as e:
        print(f"✗ Node creation test failed: {e}")
    
    try:
        test_graph.test_graph_initialization()
        print("✓ Graph initialization test passed")
    except Exception as e:
        print(f"✗ Graph initialization test failed: {e}")
    
    try:
        test_date.test_normalize_date_full()
        print("✓ Date normalization test passed")
    except Exception as e:
        print(f"✗ Date normalization test failed: {e}")
    
    try:
        test_text.test_filter_dates_numbers()
        print("✓ Text filtering test passed")
    except Exception as e:
        print(f"✗ Text filtering test failed: {e}")
    
    print("\nBasic tests completed!")
