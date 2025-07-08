# API Documentation

## Core Classes

### TemporalGraphRAG

Main class for the Temporal Graph RAG system.

```python
class TemporalGraphRAG:
    def __init__(self, start_date: str, capacity: int = 10000, 
                 similarity_threshold: float = 0.5, 
                 model_name: str = 'all-MiniLM-L6-v2')
```

#### Parameters
- `start_date`: Starting date in MM/DD/YYYY format
- `capacity`: Maximum capacity of the temporal array (default: 10000)
- `similarity_threshold`: Threshold for semantic similarity matching (default: 0.5)
- `model_name`: Sentence transformer model name (default: 'all-MiniLM-L6-v2')

#### Methods

##### `insert_news(date: str, triplets: List[Dict[str, str]])`
Insert news triplets for a specific date.

**Parameters:**
- `date`: Date string in MM/DD/YYYY format
- `triplets`: List of triplet dictionaries with keys: subject, predicate, object

**Example:**
```python
triplets = [
    {"subject": "Obama", "predicate": "visited", "object": "Chicago"},
    {"subject": "Biden", "predicate": "became", "object": "President"}
]
graph.insert_news("01/15/2020", triplets)
```

##### `search(date: str, subject_query: str, predicate_query: str, question_type: str, max_hops: int = 4, threshold: float = None)`
Search with soft matching for both subject and predicate.

**Parameters:**
- `date`: Date to search in MM/DD/YYYY format
- `subject_query`: Subject to search for
- `predicate_query`: Predicate to search for
- `question_type`: Type of question (Date, Person, Other)
- `max_hops`: Maximum hops for multi-hop search
- `threshold`: Similarity threshold (uses default if None)

**Returns:** List of matching triplets with scores

##### `search_month(month: int, year: int, subject: str, predicate_query: str, question_type: str, max_hops: int = 4, threshold: float = None)`
Search across a whole month with soft matching.

##### `answer_question(question: str)`
Answer a natural language question using the temporal graph.

**Parameters:**
- `question`: Natural language question

**Returns:** Generated answer string

##### `save(filepath: str)` / `load(filepath: str)`
Save/load the temporal graph to/from a pickle file.

##### `set_neo4j_client(uri: str, username: str, password: str)`
Set up Neo4j client for semantic search.

### Node

Represents a node in the temporal knowledge graph.

```python
class Node:
    def __init__(self, subject: str)
    def add_edge(self, predicate: str, object_: str, embedder: SentenceTransformer)
```

### QuestionProcessor

Handles question classification, entity extraction, and answer generation.

```python
class QuestionProcessor:
    def __init__(self, embedder, neo4j_client, groq_api_key: str = None)
    def classify_question(self, question: str) -> str
    def extract_entities_dates_predicate(self, question: str) -> Dict[str, Any]
    def answer_question(self, question: str, temporal_graph) -> str
```

### RAGEvaluator

Evaluator for RAG system performance using BLEU and ROUGE metrics.

```python
class RAGEvaluator:
    def evaluate_dataset(self, dataset_path: str, answer_func: Callable, ...)
    def evaluate_single(self, generated: str, ground_truth: str)
```

## Utility Functions

### Date Utils

```python
from temporal_graph_rag.utils.date_utils import normalize_date

# Normalize various date formats
result = normalize_date("March 15, 2020")
# Returns: {"normalized_date": "03/15/2020", "type": 0, "original": "March 15, 2020"}
```

### Text Utils

```python
from temporal_graph_rag.utils.text_utils import filter_triplets_with_dates_or_numbers

# Filter triplets containing dates or numbers
filtered = filter_triplets_with_dates_or_numbers(triplets)
```

## CLI Commands

### Build Graph
```bash
python -m temporal_graph_rag build [OPTIONS]
```

**Options:**
- `--input-file`: Input CSV file containing triplets and dates
- `--output-file`: Output pickle file path
- `--start-date`: Starting date (MM/DD/YYYY)
- `--date-column`: Date column name (default: Sequential_Date)
- `--triplets-column`: Triplets column name (default: triplets)

### Run Inference
```bash
python -m temporal_graph_rag infer [OPTIONS]
```

**Options:**
- `--graph-file`: Trained graph pickle file
- `--question`: Question to answer
- `--output-format`: Output format (text/json)

### Evaluate
```bash
python -m temporal_graph_rag eval [OPTIONS]
```

**Options:**
- `--graph-file`: Trained graph pickle file
- `--test-file`: Test questions CSV file
- `--output-file`: Results output file
- `--max-questions`: Maximum questions to evaluate

## Configuration

Environment variables can be set in `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=neo4j+s://your_instance.databases.neo4j.io
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
```

## Error Handling

The system includes comprehensive error handling:

- **Date parsing errors**: Invalid dates are logged and skipped
- **API failures**: Graceful degradation when services are unavailable
- **Memory limits**: Configurable capacity limits prevent memory issues
- **Validation**: Input validation for all user-provided data

## Performance Considerations

- **FAISS Integration**: Optional FAISS indexing for faster similarity search
- **Batch Processing**: Support for batch inference to handle rate limits
- **Memory Management**: Configurable graph capacity and cleanup
- **Caching**: Embedding caching to avoid recomputation
