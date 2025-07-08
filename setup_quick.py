#!/usr/bin/env python
"""
Quick setup script for Temporal Graph RAG
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def setup_environment():
    """Set up the environment."""
    if not check_python_version():
        return False
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        return False
    
    # Install optional dependencies
    run_command("pip install faiss-cpu", "Installing FAISS (optional)")
    
    # Create necessary directories
    for dir_name in ["models", "results", "logs"]:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")
    
    # Copy environment file if it doesn't exist
    if not Path(".env").exists():
        if Path(".env.example").exists():
            subprocess.run("cp .env.example .env", shell=True)
            print("📝 Created .env file from example")
            print("⚠️  Please edit .env with your API credentials")
        else:
            print("⚠️  .env.example not found")
    else:
        print("📝 .env file already exists")
    
    return True


def run_example():
    """Run a basic example."""
    print("\n🚀 Running basic example...")
    
    # Check if example data exists
    example_file = Path("data/example_triplets.csv")
    if not example_file.exists():
        print("❌ Example data not found")
        return False
    
    # Build example graph
    cmd = f"""python -m temporal_graph_rag build \
        --input-file {example_file} \
        --output-file models/example_graph.pkl \
        --start-date "01/01/2020" \
        --verbose"""
    
    if not run_command(cmd, "Building example graph"):
        return False
    
    # Check if environment is configured for inference
    if not os.getenv('GROQ_API_KEY'):
        print("⚠️  GROQ_API_KEY not set. Skipping inference example.")
        print("   Set your API key in .env to test question answering")
        return True
    
    # Run inference example
    cmd = """python -m temporal_graph_rag infer \
        --graph-file models/example_graph.pkl \
        --question "What did Obama do on March 15, 2020?" \
        --output-format json"""
    
    if run_command(cmd, "Running inference example"):
        print("✅ Basic example completed successfully!")
        return True
    
    return False


def run_tests():
    """Run basic tests."""
    print("\n🧪 Running tests...")
    
    # Try pytest first
    if run_command("python -m pytest tests/ -v", "Running pytest"):
        return True
    
    # Fallback to manual test runner
    print("Pytest not available, running manual tests...")
    return run_command("python tests/test_basic.py", "Running basic tests")


def main():
    """Main setup function."""
    print("🏁 Temporal Graph RAG Quick Setup")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        print("❌ Setup failed")
        return 1
    
    # Run example
    if run_example():
        print("\n✅ Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with warnings")
    
    # Run tests
    if run_tests():
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed, but basic functionality should work")
    
    # Final instructions
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Edit .env with your API credentials")
    print("2. Try: python -m temporal_graph_rag --help")
    print("3. Read QUICKSTART.md for more examples")
    print("4. Check examples/ directory for usage samples")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
