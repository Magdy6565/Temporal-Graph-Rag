"""
Main CLI entry point for the Temporal Graph RAG system.
"""

import click
from .build_graph import build_graph
from .inference import inference, batch_inference
from .evaluate import evaluate, evaluate_single


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Temporal Graph RAG: Time-Aware Retrieval-Augmented Generation
    
    A sophisticated RAG system that leverages temporal knowledge graphs 
    for time-aware question answering.
    
    Use --help with any command to see detailed options.
    """
    pass


# Add commands to the CLI group
cli.add_command(build_graph, name="build")
cli.add_command(inference, name="infer")
cli.add_command(batch_inference, name="batch-infer")
cli.add_command(evaluate, name="eval")
cli.add_command(evaluate_single, name="eval-single")


if __name__ == "__main__":
    cli()
