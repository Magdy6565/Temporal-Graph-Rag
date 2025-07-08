"""
CLI command for building temporal knowledge graphs from triplet data.
"""

import ast
import click
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from ..core.temporal_graph import TemporalGraphRAG


@click.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True),
              help='Input CSV file containing triplets and dates')
@click.option('--output-file', '-o', required=True, type=click.Path(),
              help='Output pickle file path for the trained graph')
@click.option('--start-date', '-s', required=True, type=str,
              help='Starting date for the temporal graph (MM/DD/YYYY)')
@click.option('--date-column', '-d', default='Sequential_Date', type=str,
              help='Name of the date column in the CSV file')
@click.option('--triplets-column', '-t', default='triplets', type=str,
              help='Name of the triplets column in the CSV file')
@click.option('--capacity', '-c', default=10000, type=int,
              help='Capacity of the temporal graph hash table')
@click.option('--similarity-threshold', default=0.5, type=float,
              help='Similarity threshold for semantic matching')
@click.option('--model-name', default='all-MiniLM-L6-v2', type=str,
              help='Sentence transformer model name')
@click.option('--invert-triplets', is_flag=True, default=False,
              help='Also insert inverted triplets (object->subject)')
@click.option('--validate-dates', is_flag=True, default=True,
              help='Validate date formats before processing')
def build_graph(input_file: str, output_file: str, start_date: str, 
                date_column: str, triplets_column: str, capacity: int,
                similarity_threshold: float, model_name: str, 
                invert_triplets: bool, validate_dates: bool):
    """
    Build a temporal knowledge graph from triplet data.
    
    This command processes a CSV file containing dated triplets and builds
    a temporal knowledge graph that can be used for time-aware question answering.
    
    Example:
        python -m temporal_graph_rag.cli.build_graph \\
            --input-file data/triplets.csv \\
            --output-file models/graph.pkl \\
            --start-date "01/01/2000"
    """
    click.echo(f"Building temporal graph from {input_file}")
    click.echo(f"Start date: {start_date}")
    click.echo(f"Output file: {output_file}")
    
    # Validate start date format
    try:
        datetime.strptime(start_date, "%m/%d/%Y")
    except ValueError:
        click.echo("Error: start-date must be in MM/DD/YYYY format", err=True)
        return
    
    # Load the dataset
    try:
        df = pd.read_csv(input_file)
        click.echo(f"Loaded dataset with {len(df)} rows")
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        return
    
    # Validate required columns
    if date_column not in df.columns:
        click.echo(f"Error: Date column '{date_column}' not found. Available columns: {list(df.columns)}", err=True)
        return
    
    if triplets_column not in df.columns:
        click.echo(f"Error: Triplets column '{triplets_column}' not found. Available columns: {list(df.columns)}", err=True)
        return
    
    # Initialize the temporal graph
    try:
        graph = TemporalGraphRAG(
            start_date=start_date,
            capacity=capacity,
            similarity_threshold=similarity_threshold,
            model_name=model_name
        )
        click.echo(f"Initialized temporal graph with capacity {capacity}")
    except Exception as e:
        click.echo(f"Error initializing graph: {e}", err=True)
        return
    
    # Process the data
    processed_count = 0
    failed_count = 0
    bad_indexes = []
    
    click.echo("Processing triplets...")
    with click.progressbar(df.iterrows(), length=len(df)) as bar:
        for index, row in bar:
            try:
                # Process date
                raw_date = row[date_column]
                if validate_dates:
                    try:
                        # Convert date format if needed
                        if '-' in str(raw_date):
                            formatted_date = datetime.strptime(str(raw_date), '%m-%d-%Y').strftime('%m/%d/%Y')
                        else:
                            formatted_date = str(raw_date)
                        
                        # Validate date format
                        datetime.strptime(formatted_date, '%m/%d/%Y')
                    except ValueError:
                        click.echo(f"Row {index}: Invalid date format '{raw_date}'")
                        bad_indexes.append(index)
                        failed_count += 1
                        continue
                else:
                    formatted_date = str(raw_date)
                
                # Process triplets
                raw_triplets = row[triplets_column]
                
                # Handle string representation of lists
                if isinstance(raw_triplets, str):
                    try:
                        triplets = ast.literal_eval(raw_triplets.strip())
                    except (ValueError, SyntaxError) as e:
                        click.echo(f"Row {index}: Failed to parse triplets - {e}")
                        bad_indexes.append(index)
                        failed_count += 1
                        continue
                else:
                    triplets = raw_triplets
                
                # Validate triplets structure
                if not isinstance(triplets, list):
                    click.echo(f"Row {index}: Triplets must be a list")
                    bad_indexes.append(index)
                    failed_count += 1
                    continue
                
                valid_triplets = []
                for triplet in triplets:
                    if (isinstance(triplet, dict) and 
                        {'subject', 'predicate', 'object'}.issubset(triplet.keys())):
                        valid_triplets.append(triplet)
                
                if not valid_triplets:
                    click.echo(f"Row {index}: No valid triplets found")
                    bad_indexes.append(index)
                    failed_count += 1
                    continue
                
                # Insert triplets
                graph.insert_news(formatted_date, valid_triplets)
                
                # Insert inverted triplets if requested
                if invert_triplets:
                    inverted_triplets = []
                    for triplet in valid_triplets:
                        inverted_triplet = {
                            'subject': triplet['object'],
                            'predicate': triplet['predicate'],
                            'object': triplet['subject']
                        }
                        inverted_triplets.append(inverted_triplet)
                    graph.insert_news(formatted_date, inverted_triplets)
                
                processed_count += 1
                
            except Exception as e:
                click.echo(f"Row {index}: Unexpected error - {e}")
                bad_indexes.append(index)
                failed_count += 1
                continue
    
    # Save the graph
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        graph.save(str(output_path))
        click.echo(f"Graph saved to {output_file}")
    except Exception as e:
        click.echo(f"Error saving graph: {e}", err=True)
        return
    
    # Summary
    click.echo("\n" + "="*50)
    click.echo("BUILD SUMMARY")
    click.echo("="*50)
    click.echo(f"Total rows processed: {len(df)}")
    click.echo(f"Successfully processed: {processed_count}")
    click.echo(f"Failed to process: {failed_count}")
    click.echo(f"Graph capacity used: {graph.size}/{graph.capacity}")
    click.echo(f"Unique subjects: {len(graph.subject_embeddings)}")
    
    if bad_indexes:
        click.echo(f"\nFailed row indexes: {bad_indexes[:10]}{'...' if len(bad_indexes) > 10 else ''}")
    
    click.echo(f"\nGraph successfully built and saved to {output_file}")


if __name__ == '__main__':
    build_graph()
