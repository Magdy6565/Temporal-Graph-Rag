"""
Temporal Graph RAG: Time-Aware Retrieval-Augmented Generation

A sophisticated RAG system that leverages temporal knowledge graphs 
for time-aware question answering.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

from .core.temporal_graph import TemporalGraphRAG, Node
from .core.question_processor import QuestionProcessor
from .core.evaluator import RAGEvaluator
from .utils.date_utils import normalize_date
from .utils.text_utils import filter_triplets_with_dates_or_numbers

__all__ = [
    "TemporalGraphRAG",
    "Node", 
    "QuestionProcessor",
    "RAGEvaluator",
    "normalize_date",
    "filter_triplets_with_dates_or_numbers",
]
