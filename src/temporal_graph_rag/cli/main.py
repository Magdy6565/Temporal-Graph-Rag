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


@cli.command()
@click.argument('legacy_file')
@click.argument('output_file')
def convert_legacy(legacy_file, output_file):
    """Convert legacy pickle file to new format."""
    from ..legacy_compat import load_legacy_pickle, convert_legacy_to_new_format
    
    try:
        click.echo(f"Loading legacy pickle file: {legacy_file}")
        legacy_obj = load_legacy_pickle(legacy_file)
        
        click.echo(f"Converting to new format...")
        convert_legacy_to_new_format(legacy_obj, output_file)
        
        click.echo(f"Successfully converted {legacy_file} to {output_file}")
        
    except Exception as e:
        click.echo(f"Error during conversion: {e}", err=True)
        click.echo("Possible solutions:", err=True)
        click.echo("1. Make sure the legacy file was created with the original notebook", err=True)
        click.echo("2. Try re-training the model with the new package", err=True)
        click.echo("3. Check file permissions and paths", err=True)


if __name__ == "__main__":
    cli()
