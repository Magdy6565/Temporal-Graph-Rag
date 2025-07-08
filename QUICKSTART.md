# Quick Start Guide

This guide will help you get started with Temporal Graph RAG quickly.

## Prerequisites

- Python 3.8 or higher
- Groq API key (for LLM operations)
- Neo4j database (for semantic search)

## Installation

```bash
# Clone and install
git clone <repository-url>
cd temporal-graph-rag
pip install -e .

# Or install from PyPI (when published)
pip install temporal-graph-rag
```

## Setup Environment

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your credentials:
```bash
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=neo4j+s://your_instance.databases.neo4j.io
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
```

## Quick Example

### 1. Prepare Data

Create a CSV file with your triplets:
```csv
Sequential_Date,triplets
03/15/2020,"[{""subject"": ""Obama"", ""predicate"": ""visited"", ""object"": ""Chicago""}]"
01/20/2021,"[{""subject"": ""Biden"", ""predicate"": ""became"", ""object"": ""President""}]"
```

### 2. Build Knowledge Graph

```bash
python -m temporal_graph_rag build \
    --input-file data/triplets.csv \
    --output-file models/my_graph.pkl \
    --start-date "01/01/2020"
```

### 3. Ask Questions

```bash
python -m temporal_graph_rag infer \
    --graph-file models/my_graph.pkl \
    --question "What did Obama do on March 15, 2020?"
```

### 4. Evaluate Performance

```bash
python -m temporal_graph_rag eval \
    --graph-file models/my_graph.pkl \
    --test-file data/test_qa.csv \
    --output-file results/evaluation.json
```

## Python API Usage

```python
from temporal_graph_rag import TemporalGraphRAG

# Initialize
graph = TemporalGraphRAG(start_date="01/01/2020")

# Add data
triplets = [{"subject": "Obama", "predicate": "visited", "object": "Chicago"}]
graph.insert_news("03/15/2020", triplets)

# Set up API connections
graph.set_neo4j_client(uri, username, password)

# Ask questions
answer = graph.answer_question("What did Obama do on March 15, 2020?")
print(answer)
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/](examples/) for more usage examples
- Run tests with `python -m pytest tests/`
- Explore the CLI help: `python -m temporal_graph_rag --help`

## Common Issues

1. **FAISS not found**: Install with `pip install faiss-cpu`
2. **Neo4j connection errors**: Check your URI and credentials
3. **Out of memory**: Reduce graph capacity or batch size
4. **API rate limits**: Increase delay between requests

## Support

- Check existing issues on GitHub
- Read the documentation
- Run the test suite to verify your installation
