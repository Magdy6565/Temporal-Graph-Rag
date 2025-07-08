"""
CLI command for evaluating temporal knowledge graph performance.
"""

import json
import click
from pathlib import Path
from ..core.temporal_graph import TemporalGraphRAG
from ..core.question_processor import QuestionProcessor
from ..core.evaluator import RAGEvaluator


@click.command()
@click.option('--graph-file', '-g', required=True, type=click.Path(exists=True),
              help='Path to the trained temporal graph pickle file')
@click.option('--test-file', '-t', required=True, type=click.Path(exists=True),
              help='CSV file with test questions and answers')
@click.option('--output-file', '-o', required=True, type=click.Path(),
              help='Output file for evaluation results (JSON format)')
@click.option('--question-column', default='question', type=str,
              help='Name of the question column in test file')
@click.option('--answer-column', default='answer', type=str,
              help='Name of the answer column in test file')
@click.option('--neo4j-uri', type=str, envvar='NEO4J_URI',
              help='Neo4j database URI')
@click.option('--neo4j-user', type=str, envvar='NEO4J_USERNAME',
              help='Neo4j username')
@click.option('--neo4j-password', type=str, envvar='NEO4J_PASSWORD',
              help='Neo4j password')
@click.option('--groq-api-key', type=str, envvar='GROQ_API_KEY',
              help='Groq API key')
@click.option('--max-questions', default=100, type=int,
              help='Maximum number of questions to evaluate')
@click.option('--delay', default=2.0, type=float,
              help='Delay between API calls in seconds')
@click.option('--save-detailed', is_flag=True, default=False,
              help='Save detailed question-answer pairs')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose output')
def evaluate(graph_file: str, test_file: str, output_file: str,
             question_column: str, answer_column: str, neo4j_uri: str,
             neo4j_user: str, neo4j_password: str, groq_api_key: str,
             max_questions: int, delay: float, save_detailed: bool, verbose: bool):
    """
    Evaluate temporal knowledge graph performance on a test dataset.
    
    This command evaluates the system using BLEU and ROUGE metrics on a
    CSV file containing question-answer pairs.
    
    Example:
        python -m temporal_graph_rag.cli.evaluate \\
            --graph-file models/graph.pkl \\
            --test-file data/test_qa.csv \\
            --output-file results/evaluation.json \\
            --max-questions 50
    """
    
    if verbose:
        click.echo(f"Loading graph from {graph_file}")
    
    # Load the temporal graph
    try:
        graph = TemporalGraphRAG.load(graph_file)
        if verbose:
            click.echo("Graph loaded successfully")
    except Exception as e:
        click.echo(f"Error loading graph: {e}", err=True)
        return
    
    # Set up Neo4j connection if credentials provided
    neo4j_configured = False
    if neo4j_uri and neo4j_user and neo4j_password:
        try:
            graph.set_neo4j_client(neo4j_uri, neo4j_user, neo4j_password)
            neo4j_configured = True
            if verbose:
                click.echo("Neo4j client configured")
        except Exception as e:
            click.echo(f"Warning: Failed to configure Neo4j: {e}")
    
    # Validate Groq API key
    if not groq_api_key:
        click.echo("Error: Groq API key is required", err=True)
        return
    
    # Initialize components
    try:
        processor = QuestionProcessor(
            embedder=graph.embedder,
            neo4j_client=graph.neo4j_client,
            groq_api_key=groq_api_key
        )
        evaluator = RAGEvaluator()
        if verbose:
            click.echo("Components initialized")
    except Exception as e:
        click.echo(f"Error initializing components: {e}", err=True)
        return
    
    # Create answer function
    def answer_func(question: str) -> str:
        try:
            return processor.answer_question(question, graph)
        except Exception as e:
            if verbose:
                click.echo(f"Error answering '{question[:50]}...': {e}")
            return ""
    
    # Prepare output paths
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    detailed_path = None
    if save_detailed:
        detailed_path = output_path.parent / f"{output_path.stem}_detailed.csv"
    
    # Run evaluation
    click.echo(f"Evaluating on {test_file} (max {max_questions} questions)")
    
    try:
        results = evaluator.evaluate_dataset(
            dataset_path=test_file,
            answer_func=answer_func,
            question_col=question_column,
            answer_col=answer_column,
            max_questions=max_questions,
            delay=delay,
            output_path=str(detailed_path) if detailed_path else None
        )
        
        if results is None:
            click.echo("Evaluation failed", err=True)
            return
        
        # Add metadata to results
        results.update({
            "graph_file": graph_file,
            "test_file": test_file,
            "neo4j_configured": neo4j_configured,
            "max_questions": max_questions,
            "question_column": question_column,
            "answer_column": answer_column,
            "similarity_threshold": graph.similarity_threshold
        })
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("EVALUATION RESULTS")
        click.echo("="*50)
        click.echo(f"Questions evaluated: {results['num_questions']}")
        click.echo(f"Average BLEU-4: {results['average_bleu4']:.4f}")
        click.echo(f"Average ROUGE-L: {results['average_rouge_l']:.4f}")
        click.echo(f"Neo4j configured: {neo4j_configured}")
        click.echo(f"Similarity threshold: {graph.similarity_threshold}")
        click.echo(f"\nResults saved to: {output_file}")
        
        if detailed_path and detailed_path.exists():
            click.echo(f"Detailed results saved to: {detailed_path}")
        
    except Exception as e:
        click.echo(f"Error during evaluation: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)


@click.command()
@click.option('--question', '-q', required=True, type=str,
              help='Question to evaluate')
@click.option('--answer', '-a', required=True, type=str,
              help='Ground truth answer')
@click.option('--graph-file', '-g', required=True, type=click.Path(exists=True),
              help='Path to the trained temporal graph pickle file')
@click.option('--groq-api-key', type=str, envvar='GROQ_API_KEY',
              help='Groq API key')
@click.option('--neo4j-uri', type=str, envvar='NEO4J_URI',
              help='Neo4j database URI')
@click.option('--neo4j-user', type=str, envvar='NEO4J_USERNAME',
              help='Neo4j username')
@click.option('--neo4j-password', type=str, envvar='NEO4J_PASSWORD',
              help='Neo4j password')
def evaluate_single(question: str, answer: str, graph_file: str,
                   groq_api_key: str, neo4j_uri: str, neo4j_user: str,
                   neo4j_password: str):
    """
    Evaluate a single question-answer pair.
    
    Example:
        python -m temporal_graph_rag.cli.evaluate evaluate-single \\
            --question "What happened on March 15, 2018?" \\
            --answer "Obama visited Chicago" \\
            --graph-file models/graph.pkl
    """
    
    # Load components
    graph = TemporalGraphRAG.load(graph_file)
    
    if neo4j_uri and neo4j_user and neo4j_password:
        graph.set_neo4j_client(neo4j_uri, neo4j_user, neo4j_password)
    
    processor = QuestionProcessor(
        embedder=graph.embedder,
        neo4j_client=graph.neo4j_client,
        groq_api_key=groq_api_key
    )
    
    evaluator = RAGEvaluator()
    
    # Generate answer
    try:
        generated_answer = processor.answer_question(question, graph)
    except Exception as e:
        click.echo(f"Error generating answer: {e}", err=True)
        return
    
    # Evaluate
    metrics = evaluator.evaluate_single(generated_answer, answer)
    
    # Display results
    click.echo(f"Question: {question}")
    click.echo(f"Generated: {generated_answer}")
    click.echo(f"Ground Truth: {answer}")
    click.echo(f"BLEU-4: {metrics['bleu4']:.4f}")
    click.echo(f"ROUGE-L: {metrics['rouge_l']:.4f}")


if __name__ == '__main__':
    evaluate()
