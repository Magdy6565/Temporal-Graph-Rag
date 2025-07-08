# Temporal Graph RAG: Time-Aware Retrieval-Augmented Generation

A sophisticated Retrieval-Augmented Generation (RAG) system that leverages temporal knowledge graphs for time-aware question answering. This system combines manual graph structures for precise temporal queries with Neo4j semantic search for broader contextual understanding.

## Features

- **Temporal Graph Construction**: Build time-indexed knowledge graphs from triplet data
- **Multi-Modal Search Strategy**: 
  - Precise date-based queries (MM/DD/YYYY)
  - Month-range queries (MM/YYYY)
  - Year-based semantic search
  - General semantic search via Neo4j
- **Semantic Similarity**: Advanced embedding-based entity and predicate matching
- **Question Classification**: Automatic categorization of questions by type (Date, Person, Other)
- **Performance Evaluation**: Built-in BLEU and ROUGE scoring for answer quality assessment

## Architecture

The system implements a hybrid approach:

1. **Manual Temporal Graph**: Hash-based storage for precise temporal queries
2. **Neo4j Integration**: Semantic search for broader contextual queries
3. **LLM Integration**: Groq API for entity extraction and answer generation
4. **Embedding Models**: Sentence-BERT for semantic similarity

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd temporal-graph-rag

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Environment Setup

Create a `.env` file with your credentials:

```bash
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
```

### 2. Build Knowledge Graph

```bash
# Build temporal graph from triplets
python -m temporal_graph_rag.build_graph \
    --input-file data/triplets.csv \
    --output-file models/knowledge_graph.pkl \
    --start-date "01/01/2000"
```

### 3. Run Inference

```bash
# Answer questions using the temporal graph
python -m temporal_graph_rag.inference \
    --graph-file models/knowledge_graph.pkl \
    --question "What happened on March 15, 2018?" \
    --output-format json
```

### 4. Evaluate Performance

```bash
# Evaluate on test dataset
python -m temporal_graph_rag.evaluate \
    --graph-file models/knowledge_graph.pkl \
    --test-file data/test_questions.csv \
    --output-file results/evaluation.json
```

## Usage Examples

### Building a Knowledge Graph

```python
from temporal_graph_rag import TemporalGraphRAG

# Initialize the system
rag = TemporalGraphRAG(start_date="01/01/2000")

# Load triplets and build graph
triplets = [
    {"subject": "Obama", "predicate": "became", "object": "President"},
    # ... more triplets
]
rag.insert_news("01/20/2009", triplets)

# Save the graph
rag.save("knowledge_graph.pkl")
```

### Querying the System

```python
# Load existing graph
rag = TemporalGraphRAG.load("knowledge_graph.pkl")

# Ask questions
answer = rag.answer_question("Who became President on January 20, 2009?")
print(answer)
```

## API Reference

### Core Classes

#### `TemporalGraphRAG`

Main class for temporal graph operations.

**Methods:**
- `__init__(start_date: str)`: Initialize with starting date
- `insert_news(date: str, triplets: List[Dict])`: Add triplets for a specific date
- `search(date: str, subject: str, predicate: str, question_type: str)`: Search for specific patterns
- `answer_question(question: str)`: Generate answer for natural language question
- `save(filepath: str)`: Serialize graph to file
- `load(filepath: str)`: Load graph from file

#### `Node`

Represents entities in the knowledge graph.

**Attributes:**
- `subject`: Entity identifier
- `edges`: Dictionary of predicates to objects
- `predicate_embeddings`: Cached embeddings for predicates

### CLI Commands

#### Build Graph
```bash
python -m temporal_graph_rag.build_graph [OPTIONS]

Options:
  --input-file PATH          Input CSV file with triplets [required]
  --output-file PATH         Output pickle file path [required]
  --start-date TEXT          Starting date (MM/DD/YYYY) [required]
  --date-column TEXT         Date column name [default: Sequential_Date]
  --triplets-column TEXT     Triplets column name [default: triplets]
```

#### Run Inference
```bash
python -m temporal_graph_rag.inference [OPTIONS]

Options:
  --graph-file PATH          Trained graph pickle file [required]
  --question TEXT            Question to answer [required]
  --neo4j-uri TEXT           Neo4j database URI
  --neo4j-user TEXT          Neo4j username
  --neo4j-password TEXT      Neo4j password
  --groq-api-key TEXT        Groq API key
  --output-format TEXT       Output format (text/json) [default: text]
```

#### Evaluate Model
```bash
python -m temporal_graph_rag.evaluate [OPTIONS]

Options:
  --graph-file PATH          Trained graph pickle file [required]
  --test-file PATH           Test questions CSV file [required]
  --output-file PATH         Results output file [required]
  --question-column TEXT     Question column name [default: question]
  --answer-column TEXT       Answer column name [default: answer]
  --max-questions INTEGER    Maximum questions to evaluate [default: 100]
```

## Data Format

### Input Triplets CSV

```csv
Sequential_Date,triplets
01/15/2018,"[{""subject"": ""Obama"", ""predicate"": ""visited"", ""object"": ""Chicago""}]"
01/16/2018,"[{""subject"": ""Tesla"", ""predicate"": ""announced"", ""object"": ""new model""}]"
```

### Test Questions CSV

```csv
question,answer
"What did Obama do on January 15, 2018?","Obama visited Chicago"
"What did Tesla announce on January 16, 2018?","Tesla announced new model"
```

## Performance

The system handles different query types with varying strategies:

- **Exact Date Queries** (MM/DD/YYYY): Direct hash lookup - O(1)
- **Month Queries** (MM/YYYY): Iterative daily search - O(days_in_month)
- **Year/General Queries**: Neo4j semantic search - O(log n)

## Troubleshooting

### Legacy Pickle Files

If you get an error like `Can't get attribute 'Hash' on module '__main__'`, you're trying to load a pickle file created with the original notebook code. Here are solutions:

**Option 1: Convert using CLI**
```bash
python -m temporal_graph_rag convert-legacy old_file.pkl new_file.pkl
```

**Option 2: For Kaggle users**
Download the standalone fixer:
```python
# In Kaggle
import requests
url = "https://raw.githubusercontent.com/your-repo/temporal-graph-rag/main/kaggle_pickle_fix.py"
response = requests.get(url)
with open("kaggle_pickle_fix.py", "w") as f:
    f.write(response.text)

# Fix your legacy file
exec(open("kaggle_pickle_fix.py").read())
kaggle_quick_fix("/kaggle/input/your-dataset/model.pkl", "fixed_model.pkl")
```

**Option 3: Re-build from scratch (recommended)**
```bash
python -m temporal_graph_rag build --input-file your_triplets.json --output-file new_graph.pkl
```

See `PICKLE_FIX_GUIDE.md` for detailed instructions.

### Other Common Issues

- **Import Errors**: Make sure you've installed with `pip install -e .`
- **API Key Issues**: Check your `.env` file and API key validity
- **Memory Issues**: For large graphs, consider using FAISS for embedding storage
- **Neo4j Connection**: Ensure your Neo4j instance is running and credentials are correct

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{temporal_graph_rag_2024,
  title={Temporal Graph RAG: Time-Aware Retrieval-Augmented Generation},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Sentence-BERT](https://www.sbert.net/) for semantic embeddings
- [Neo4j](https://neo4j.com/) for graph database operations
- [Groq](https://groq.com/) for LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
