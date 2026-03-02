"""
Evaluation metrics for fine-tuned models.
Implements perplexity, BLEU, ROUGE, and sample generation.
"""

import math
import os
from typing import List, Dict, Any, Optional, Union

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset


class Evaluator:
    """Compute evaluation metrics for language models."""
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        
    def set_model(self, model, tokenizer):
        """Set or update the model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
    
    def compute_perplexity(
        self,
        dataset: Dataset,
        max_samples: int = None,
        batch_size: int = 1,
        text_column: str = "text"
    ) -> float:
        """
        Calculate perplexity on a dataset.
        
        Args:
            dataset: Dataset with text samples
            max_samples: Maximum number of samples to evaluate (None for all)
            batch_size: Batch size for evaluation
            text_column: Column containing text
            
        Returns:
            Perplexity score (lower is better)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set before evaluation")
        
        self.model.eval()
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), batch_size), desc="Computing perplexity"):
                batch = dataset[i:i+batch_size]
                texts = batch[text_column] if isinstance(batch[text_column], list) else [batch[text_column]]
                
                # Tokenize
                encodings = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length
                )
                
                input_ids = encodings.input_ids.to(self.model.device)
                attention_mask = encodings.attention_mask.to(self.model.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # Count valid tokens (excluding padding)
                valid_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings, or list of lists for multiple references
            
        Returns:
            Dictionary with BLEU scores
        """
        try:
            import sacrebleu
            
            # sacrebleu expects list of lists for references
            if references and not isinstance(references[0], list):
                references = [[ref] for ref in references]
            
            bleu = sacrebleu.corpus_bleu(predictions, list(zip(*references)))
            
            return {
                "bleu": bleu.score,
                "bleu_1": bleu.precisions[0],
                "bleu_2": bleu.precisions[1],
                "bleu_3": bleu.precisions[2],
                "bleu_4": bleu.precisions[3],
            }
        except ImportError:
            # Fallback to nltk
            try:
                from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
                
                # Tokenize
                pred_tokens = [p.split() for p in predictions]
                
                if references and isinstance(references[0], str):
                    ref_tokens = [[r.split()] for r in references]
                else:
                    ref_tokens = [[r.split() for r in ref_list] for ref_list in references]
                
                smoothing = SmoothingFunction()
                
                return {
                    "bleu": corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing.method1) * 100,
                    "bleu_method": "nltk"
                }
            except ImportError:
                raise ImportError("Please install sacrebleu or nltk for BLEU calculation: pip install sacrebleu")
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            return {
                "rouge1": np.mean(rouge1_scores) * 100,
                "rouge2": np.mean(rouge2_scores) * 100,
                "rougeL": np.mean(rougeL_scores) * 100,
            }
        except ImportError:
            raise ImportError("Please install rouge-score: pip install rouge-score")
    
    def generate_samples(
        self,
        dataset: Dataset,
        num_samples: int = 10,
        max_new_tokens: int = 256,
        text_column: str = "text",
        extract_prompt: bool = True
    ) -> List[Dict[str, str]]:
        """
        Generate predictions for sample comparison.
        
        Args:
            dataset: Dataset with text samples
            num_samples: Number of samples to generate
            max_new_tokens: Maximum tokens to generate
            text_column: Column containing text
            extract_prompt: Try to extract prompt from formatted text
            
        Returns:
            List of dicts with input, expected_output, generated_output
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set before generation")
        
        self.model.eval()
        
        # Sample from dataset
        if num_samples >= len(dataset):
            samples = dataset
        else:
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            samples = dataset.select(indices.tolist())
        
        results = []
        
        with torch.no_grad():
            for i in tqdm(range(len(samples)), desc="Generating samples"):
                example = samples[i]
                full_text = example[text_column]
                
                # Try to extract prompt (everything before output)
                if extract_prompt and "<|endoftext|>" in full_text:
                    # Remove the end token and split
                    text_without_eot = full_text.replace("<|endoftext|>", "")
                    
                    # Common patterns
                    if "### Response:" in text_without_eot:
                        parts = text_without_eot.split("### Response:")
                        prompt = parts[0] + "### Response:"
                        expected = parts[1].strip() if len(parts) > 1 else ""
                    elif "<|im_start|>assistant" in text_without_eot:
                        parts = text_without_eot.split("<|im_start|>assistant")
                        prompt = parts[0] + "<|im_start|>assistant\n"
                        expected = parts[1].replace(" ", "").strip() if len(parts) > 1 else ""
                    else:
                        # Use first half as prompt
                        mid = len(text_without_eot) // 2
                        prompt = text_without_eot[:mid]
                        expected = text_without_eot[mid:]
                else:
                    prompt = full_text
                    expected = ""
                
                # Generate
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                results.append({
                    "input": prompt,
                    "expected": expected,
                    "generated": generated,
                })
        
        return results
    
    def evaluate_all(
        self,
        test_dataset: Dataset,
        metrics: List[str] = ["perplexity"],
        num_samples: int = 100,
        generate_report: bool = False,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Run all requested evaluations.
        
        Args:
            test_dataset: Test dataset
            metrics: List of metrics to compute
            num_samples: Number of samples for evaluation
            generate_report: Whether to generate HTML report
            output_path: Path for report output
            
        Returns:
            Dictionary with all computed metrics
        """
        results = {}
        
        if "perplexity" in metrics:
            print("Computing perplexity...")
            results["perplexity"] = self.compute_perplexity(test_dataset, max_samples=num_samples)
        
        # For BLEU and ROUGE, we need to generate predictions
        if "bleu" in metrics or "rouge" in metrics:
            print("Generating samples for BLEU/ROUGE...")
            samples = self.generate_samples(test_dataset, num_samples=min(num_samples, 50))
            
            predictions = [s["generated"] for s in samples]
            references = [s["expected"] for s in samples]
            
            if "bleu" in metrics:
                print("Computing BLEU...")
                results["bleu"] = self.compute_bleu(predictions, references)
            
            if "rouge" in metrics:
                print("Computing ROUGE...")
                results["rouge"] = self.compute_rouge(predictions, references)
        
        # Generate report if requested
        if generate_report:
            if output_path is None:
                output_path = "evaluation_report.html"
            self.create_report(results, samples if 'samples' in locals() else [], output_path)
            results["report_path"] = output_path
        
        return results
    
    def create_report(
        self,
        metrics: Dict[str, Any],
        samples: List[Dict[str, str]],
        output_path: str
    ):
        """
        Generate HTML report with metrics and sample predictions.
        
        Args:
            metrics: Dictionary of computed metrics
            samples: List of sample predictions
            output_path: Path for HTML output
        """
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .sample { margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #4CAF50; }
        .input { color: #666; font-style: italic; margin: 10px 0; }
        .expected { color: #2196F3; margin: 10px 0; }
        .generated { color: #4CAF50; margin: 10px 0; }
        pre { white-space: pre-wrap; word-wrap: break-word; background: #f4f4f4; padding: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Model Evaluation Report</h1>
"""
        
        # Metrics section
        html += "    <h2>Metrics</h2>\n    <table>\n"
        html += "        <tr><th>Metric</th><th>Value</th></tr>\n"
        
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    html += f"        <tr><td>{metric_name} - {sub_name}</td><td>{sub_value:.4f}</td></tr>\n"
            else:
                html += f"        <tr><td>{metric_name}</td><td>{value:.4f}</td></tr>\n"
        
        html += "    </table>\n"
        
        # Samples section
        if samples:
            html += "    <h2>Sample Predictions</h2>\n"
            
            for i, sample in enumerate(samples[:10], 1):  # Show first 10
                html += f"    <div class='sample'>\n"
                html += f"        <h3>Sample {i}</h3>\n"
                html += f"        <div class='input'><strong>Input:</strong><pre>{sample['input'][:500]}...</pre></div>\n"
                html += f"        <div class='expected'><strong>Expected:</strong><pre>{sample['expected'][:500]}</pre></div>\n"
                html += f"        <div class='generated'><strong>Generated:</strong><pre>{sample['generated'][:500]}</pre></div>\n"
                html += f"    </div>\n"
        
        html += "</body>\n</html>"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Report saved to: {output_path}")
