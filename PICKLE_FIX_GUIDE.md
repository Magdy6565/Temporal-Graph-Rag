# Fixing Legacy Pickle Loading Issues

## Problem
When you see an error like:
```
Can't get attribute 'Hash' on module '__main__'
```

This happens because you're trying to load a pickle file that was created with the original notebook code, but the new package structure doesn't have the same class definitions.

## Solution Options

### Option 1: Convert Legacy Pickle to New Format

1. **Using the CLI command:**
```bash
cd temporal-graph-rag
python -m temporal_graph_rag convert-legacy your_old_file.pkl new_file.pkl
```

2. **Using Python script:**
```python
from temporal_graph_rag.legacy_compat import load_legacy_pickle, convert_legacy_to_new_format

# Load the legacy file
legacy_obj = load_legacy_pickle("your_old_file.pkl")

# Convert to new format
convert_legacy_to_new_format(legacy_obj, "new_file.pkl")
```

### Option 2: Register Legacy Classes (Temporary Fix)

If you need a quick fix without converting:

```python
from temporal_graph_rag.legacy_compat import register_legacy_classes
import pickle

# Register legacy classes first
register_legacy_classes()

# Now load the pickle file
with open("your_file.pkl", "rb") as f:
    data = pickle.load(f)
```

### Option 3: Re-train from Scratch (Recommended)

The cleanest solution is to re-build your knowledge graph using the new package:

```bash
# Build a new graph from your triplets data
python -m temporal_graph_rag build --input-file your_triplets.json --output-file new_graph.pkl

# Or use the legacy CLI interface
python graph_rag_cli.py --question "your question" --kg-file new_graph.pkl --groq-key YOUR_KEY
```

## For Kaggle Users

If you're running on Kaggle and get this error:

1. **Download the legacy compatibility module:**
```python
import requests

# Download the legacy compatibility module
url = "https://raw.githubusercontent.com/your-repo/temporal-graph-rag/main/src/temporal_graph_rag/legacy_compat.py"
response = requests.get(url)

with open("legacy_compat.py", "w") as f:
    f.write(response.text)
```

2. **Use it to convert your file:**
```python
exec(open("legacy_compat.py").read())

# Register classes and load
register_legacy_classes()
legacy_obj = load_legacy_pickle("/kaggle/input/your-dataset/model.pkl")

# Convert to new format
convert_legacy_to_new_format(legacy_obj, "new_model.pkl")
```

## Technical Details

The issue occurs because:
- The original notebook defined classes like `Hash` in the `__main__` module
- When pickle saves objects, it stores the module path (`__main__.Hash`)
- The new package structure doesn't have these classes in `__main__`
- The compatibility module recreates these classes and registers them properly

## Prevention

To avoid this issue in the future:
- Always use the new package structure for new models
- Save models using `TemporalGraphRAG.save()` method
- Keep backups of your training data so you can rebuild if needed
