#!/usr/bin/env python3
"""
Test script to verify legacy pickle compatibility.

This script creates a mock legacy pickle file and tests the conversion process.
"""

import pickle
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_mock_legacy_pickle():
    """Create a mock legacy pickle file for testing."""
    
    # Import after adding to path
    from temporal_graph_rag.legacy_compat import Hash
    
    # Create a mock Hash object like in the original notebook
    hash_obj = Hash("01/01/2020")
    
    # Add some test data
    test_data = [
        {
            'subject': 'Apple Inc',
            'predicate': 'announced',
            'object': 'iPhone 15'
        },
        {
            'subject': 'Apple Inc',
            'predicate': 'launched',
            'object': 'Vision Pro'
        }
    ]
    
    hash_obj.insert_news("01/15/2020", test_data)
    
    # Save as pickle
    test_file = "test_legacy.pkl"
    with open(test_file, 'wb') as f:
        pickle.dump(hash_obj, f)
    
    print(f"Created mock legacy pickle file: {test_file}")
    return test_file


def test_legacy_loading():
    """Test loading legacy pickle files."""
    print("=== Testing Legacy Pickle Loading ===")
    
    # Create mock file
    test_file = create_mock_legacy_pickle()
    
    try:
        # Test the compatibility loading
        from temporal_graph_rag.legacy_compat import load_legacy_pickle, convert_legacy_to_new_format
        
        print("1. Testing legacy pickle loading...")
        legacy_obj = load_legacy_pickle(test_file)
        print("✅ Successfully loaded legacy pickle!")
        
        print("2. Testing conversion to new format...")
        new_file = "test_converted.pkl"
        convert_legacy_to_new_format(legacy_obj, new_file)
        print("✅ Successfully converted to new format!")
        
        print("3. Testing new format loading...")
        from temporal_graph_rag.core.temporal_graph import TemporalGraphRAG
        new_graph = TemporalGraphRAG.load(new_file)
        print("✅ Successfully loaded converted file!")
        
        print("4. Testing search functionality...")
        results = new_graph.search("01/15/2020", subject="Apple Inc")
        print(f"✅ Search returned {len(results)} results")
        
        # Cleanup
        os.remove(test_file)
        os.remove(new_file)
        print("🧹 Cleaned up test files")
        
        print("\n🎉 All tests passed! Legacy compatibility is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on failure
        for f in [test_file, "test_converted.pkl"]:
            if os.path.exists(f):
                os.remove(f)


def test_cli_conversion():
    """Test CLI conversion command."""
    print("\n=== Testing CLI Conversion ===")
    
    # Create mock file
    test_file = create_mock_legacy_pickle()
    
    try:
        # Test CLI command
        import subprocess
        cmd = [
            sys.executable, "-m", "temporal_graph_rag", 
            "convert-legacy", test_file, "cli_converted.pkl"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ CLI conversion successful!")
            print("Output:", result.stdout)
        else:
            print("❌ CLI conversion failed!")
            print("Error:", result.stderr)
        
        # Cleanup
        for f in [test_file, "cli_converted.pkl"]:
            if os.path.exists(f):
                os.remove(f)
                
    except Exception as e:
        print(f"❌ CLI test failed: {e}")


if __name__ == "__main__":
    print("Testing Legacy Pickle Compatibility")
    print("=" * 40)
    
    test_legacy_loading()
    test_cli_conversion()
    
    print("\n✨ Testing complete!")
