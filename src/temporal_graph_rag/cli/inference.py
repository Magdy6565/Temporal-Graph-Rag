"""
CLI command for running inference on temporal knowledge graphs.
"""

import json
import os
import click
from pathlib import Path
from ..core.temporal_graph import TemporalGraphRAG
from ..core.question_processor import QuestionProcessor


@click.command()
@click.option('--graph-file', '-g', required=True, type=click.Path(exists=True),
              help='Path to the trained temporal graph pickle file')
@click.option('--question', '-q', required=True, type=str,
              help='Question to answer')
@click.option('--neo4j-uri', type=str, envvar='NEO4J_URI',
              help='Neo4j database URI (can be set via NEO4J_URI env var)')
@click.option('--neo4j-user', type=str, envvar='NEO4J_USERNAME',
              help='Neo4j username (can be set via NEO4J_USERNAME env var)')
@click.option('--neo4j-password', type=str, envvar='NEO4J_PASSWORD',
              help='Neo4j password (can be set via NEO4J_PASSWORD env var)')
@click.option('--groq-api-key', type=str, envvar='GROQ_API_KEY',
              help='Groq API key (can be set via GROQ_API_KEY env var)')
@click.option('--output-format', '-f', default='text', type=click.Choice(['text', 'json']),
              help='Output format: text or json')
@click.option('--similarity-threshold', default=None, type=float,
              help='Override similarity threshold for search')
@click.option('--max-hops', default=4, type=int,
              help='Maximum hops for multi-hop search')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose output')
def inference(graph_file: str, question: str, neo4j_uri: str, neo4j_user: str,
              neo4j_password: str, groq_api_key: str, output_format: str,
              similarity_threshold: float, max_hops: int, verbose: bool):
    """
    Run inference on a temporal knowledge graph.
    
    This command loads a pre-trained temporal graph and answers questions
    using a combination of temporal search and semantic search via Neo4j.
    
    Environment variables can be used for credentials:
    - NEO4J_URI
    - NEO4J_USERNAME  
    - NEO4J_PASSWORD
    - GROQ_API_KEY
    
    Example:
        python -m temporal_graph_rag.cli.inference \\
            --graph-file models/graph.pkl \\
            --question "What happened on March 15, 2018?" \\
            --output-format json
    """
    
    if verbose:
        click.echo(f"Loading graph from {graph_file}")
    
    # Load the temporal graph
    try:
        graph = TemporalGraphRAG.load(graph_file)
        if verbose:
            click.echo(f"Graph loaded successfully")
            click.echo(f"Graph capacity: {graph.size}/{graph.capacity}")
            click.echo(f"Unique subjects: {len(graph.subject_embeddings)}")
    except Exception as e:
        click.echo(f"Error loading graph: {e}", err=True)
        return
    
    # Override similarity threshold if provided
    if similarity_threshold is not None:
        graph.similarity_threshold = similarity_threshold
        if verbose:
            click.echo(f"Similarity threshold set to {similarity_threshold}")
    
    # Set up Neo4j connection if credentials provided
    neo4j_configured = False
    if neo4j_uri and neo4j_user and neo4j_password:
        try:
            graph.set_neo4j_client(neo4j_uri, neo4j_user, neo4j_password)
            neo4j_configured = True
            if verbose:
                click.echo("Neo4j client configured successfully")
        except Exception as e:
            click.echo(f"Warning: Failed to configure Neo4j client: {e}")
            if verbose:
                click.echo("Continuing without Neo4j support...")
    elif verbose:
        click.echo("Neo4j credentials not provided, semantic search will be limited")
    
    # Validate Groq API key
    if not groq_api_key:
        click.echo("Error: Groq API key is required for question processing", err=True)
        click.echo("Set GROQ_API_KEY environment variable or use --groq-api-key option")
        return
    
    # Initialize question processor
    try:
        processor = QuestionProcessor(
            embedder=graph.embedder,
            neo4j_client=graph.neo4j_client,
            groq_api_key=groq_api_key
        )
        if verbose:
            click.echo("Question processor initialized")
    except Exception as e:
        click.echo(f"Error initializing question processor: {e}", err=True)
        return
    
    # Process the question
    if verbose:
        click.echo(f"Processing question: {question}")
    
    try:
        # Get question type and extracted info for verbose output
        if verbose:
            question_type = processor.classify_question(question)
            extracted = processor.extract_entities_dates_predicate(question)
            click.echo(f"Question type: {question_type}")
            click.echo(f"Extracted entities: {extracted.get('entities', [])}")
            click.echo(f"Extracted dates: {[d.get('original') for d in extracted.get('dates', [])]}")
            click.echo(f"Extracted predicate: {extracted.get('predicate')}")
        
        # Generate answer
        answer = processor.answer_question(question, graph)
        
        # Format output
        if output_format == 'json':
            result = {
                "question": question,
                "answer": answer,
                "graph_file": graph_file,
                "neo4j_configured": neo4j_configured,
                "similarity_threshold": graph.similarity_threshold
            }
            
            if verbose:
                result.update({
                    "question_type": question_type if verbose else None,
                    "extracted_info": extracted if verbose else None
                })
            
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Question: {question}")
            click.echo(f"Answer: {answer}")
            
    except Exception as e:
        error_msg = f"Error processing question: {e}"
        
        if output_format == 'json':
            result = {
                "question": question,
                "error": str(e),
                "answer": None
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(error_msg, err=True)
        
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)


@click.command()
@click.option('--graph-file', '-g', required=True, type=click.Path(exists=True),
              help='Path to the trained temporal graph pickle file')
@click.option('--questions-file', '-q', type=click.Path(exists=True),
              help='Text file with questions (one per line)')
@click.option('--neo4j-uri', type=str, envvar='NEO4J_URI',
              help='Neo4j database URI')
@click.option('--neo4j-user', type=str, envvar='NEO4J_USERNAME', 
              help='Neo4j username')
@click.option('--neo4j-password', type=str, envvar='NEO4J_PASSWORD',
              help='Neo4j password')
@click.option('--groq-api-key', type=str, envvar='GROQ_API_KEY',
              help='Groq API key')
@click.option('--output-file', '-o', type=click.Path(),
              help='Output file for results (JSON format)')
@click.option('--delay', default=1.0, type=float,
              help='Delay between questions in seconds')
def batch_inference(graph_file: str, questions_file: str, neo4j_uri: str,
                   neo4j_user: str, neo4j_password: str, groq_api_key: str,
                   output_file: str, delay: float):
    """
    Run batch inference on multiple questions.
    
    Example:
        python -m temporal_graph_rag.cli.inference batch-inference \\
            --graph-file models/graph.pkl \\
            --questions-file data/questions.txt \\
            --output-file results/answers.json
    """
    import time
    
    # Load graph
    graph = TemporalGraphRAG.load(graph_file)
    
    # Set up connections
    if neo4j_uri and neo4j_user and neo4j_password:
        graph.set_neo4j_client(neo4j_uri, neo4j_user, neo4j_password)
    
    processor = QuestionProcessor(
        embedder=graph.embedder,
        neo4j_client=graph.neo4j_client,
        groq_api_key=groq_api_key
    )
    
    # Load questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    click.echo(f"Processing {len(questions)} questions...")
    
    results = []
    with click.progressbar(questions) as bar:
        for question in bar:
            try:
                if delay > 0:
                    time.sleep(delay)
                answer = processor.answer_question(question, graph)
                results.append({
                    "question": question,
                    "answer": answer,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "answer": None,
                    "error": str(e)
                })
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        click.echo(f"Results saved to {output_file}")
    else:
        for result in results:
            click.echo(json.dumps(result, indent=2))


if __name__ == '__main__':
    inference()
