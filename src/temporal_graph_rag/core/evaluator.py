"""
Evaluation utilities for RAG system performance assessment.
"""

import time
import pandas as pd
import nltk
from typing import List, Dict, Any, Callable
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    nltk.download('punkt')


class RAGEvaluator:
    """Evaluator for RAG system performance using BLEU and ROUGE metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        
    def evaluate_dataset(self, dataset_path: str, answer_func: Callable[[str], str],
                        question_col: str = 'question', answer_col: str = 'answer',
                        max_questions: int = 100, delay: float = 2.0,
                        output_path: str = None) -> Dict[str, float]:
        """
        Evaluate RAG system on a dataset.
        
        Args:
            dataset_path: Path to CSV dataset
            answer_func: Function that takes question and returns answer
            question_col: Name of question column
            answer_col: Name of answer column
            max_questions: Maximum questions to evaluate
            delay: Delay between API calls (seconds)
            output_path: Path to save detailed results
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Loading dataset: {dataset_path}")
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"Error: Dataset not found at {dataset_path}")
            return None

        # Validate columns
        if question_col not in df.columns or answer_col not in df.columns:
            print(f"Error: Required columns not found. Available: {df.columns.tolist()}")
            return None

        generated_answers = []
        ground_truth_answers = []
        questions = []
        
        print(f"Processing up to {min(len(df), max_questions)} questions...")
        
        # Process questions
        for index, row in df.iterrows():
            if len(questions) >= max_questions:
                break
                
            question = row[question_col]
            ground_truth = str(row[answer_col])
            questions.append(question)
            ground_truth_answers.append(ground_truth)

            # Generate answer
            try:
                if delay > 0:
                    time.sleep(delay)
                generated_answer = str(answer_func(question))
                generated_answers.append(generated_answer)
                print(f"Processed {len(questions)}/{min(len(df), max_questions)}")
            except Exception as e:
                print(f"Error generating answer for question: {question[:50]}... - {e}")
                generated_answers.append("")

        # Save detailed results if requested
        if output_path:
            results_df = pd.DataFrame({
                "Question": questions,
                "Generated Answer": generated_answers,
                "Ground Truth": ground_truth_answers
            })
            results_df.to_csv(output_path, index=False)
            print(f"Detailed results saved to {output_path}")

        # Compute metrics
        return self._compute_metrics(generated_answers, ground_truth_answers)
    
    def _compute_metrics(self, generated: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Compute BLEU and ROUGE metrics.
        
        Args:
            generated: List of generated answers
            ground_truth: List of ground truth answers
            
        Returns:
            Dictionary with computed metrics
        """
        # BLEU-4 Score
        bleu4_scores = []
        for ref, gen in zip(ground_truth, generated):
            reference_tokens = [nltk.word_tokenize(ref)]
            candidate_tokens = nltk.word_tokenize(gen)
            score = sentence_bleu(
                reference_tokens, candidate_tokens, 
                weights=(0.25, 0.25, 0.25, 0.25), 
                smoothing_function=self.smoothie
            )
            bleu4_scores.append(score)

        average_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0

        # ROUGE-L Score
        rougeL_scores = []
        for ref, gen in zip(ground_truth, generated):
            scores = self.scorer.score(ref, gen)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        average_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

        return {
            "average_bleu4": average_bleu4,
            "average_rouge_l": average_rougeL,
            "num_questions": len(generated)
        }
    
    def evaluate_single(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair.
        
        Args:
            generated: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with metrics for this pair
        """
        # BLEU-4
        reference_tokens = [nltk.word_tokenize(ground_truth)]
        candidate_tokens = nltk.word_tokenize(generated)
        bleu4 = sentence_bleu(
            reference_tokens, candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothie
        )
        
        # ROUGE-L
        rouge_scores = self.scorer.score(ground_truth, generated)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        return {
            "bleu4": bleu4,
            "rouge_l": rouge_l
        }
