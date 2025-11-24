#!/usr/bin/env python3
"""
Shield Comparison Test Framework

Compares Llama Guard 3 8B with Llama 3 8B Instruct for safety classification performance.
"""

import argparse
import json
import time
from typing import Dict, List, Tuple
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd

from model_clients import LlamaGuardClient, LlamaInstructClient
from test_dataset import TEST_CASES, SAFETY_CATEGORIES, get_test_cases_by_safety


console = Console()
PASSWORD = "h3110w0r1d"


class TestResult:
    """Container for test results."""

    def __init__(self, test_id: str, prompt: str, expected_safe: bool, expected_categories: List[str]):
        self.test_id = test_id
        self.prompt = prompt
        self.expected_safe = expected_safe
        self.expected_categories = expected_categories

        # Guard results
        self.guard_safe = None
        self.guard_categories = []
        self.guard_raw = ""
        self.guard_error = None
        self.guard_time = 0.0

        # Instruct results
        self.instruct_safe = None
        self.instruct_categories = []
        self.instruct_raw = ""
        self.instruct_error = None
        self.instruct_time = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "prompt": self.prompt,
            "expected": {
                "safe": self.expected_safe,
                "categories": self.expected_categories
            },
            "guard": {
                "safe": self.guard_safe,
                "categories": self.guard_categories,
                "raw_response": self.guard_raw,
                "error": self.guard_error,
                "time_seconds": self.guard_time
            },
            "instruct": {
                "safe": self.instruct_safe,
                "categories": self.instruct_categories,
                "raw_response": self.instruct_raw,
                "error": self.instruct_error,
                "time_seconds": self.instruct_time
            }
        }


def run_guard_test(client: LlamaGuardClient, test_case) -> Tuple[Dict, float]:
    """Run a single test with Llama Guard."""
    start = time.time()
    result = client.classify_input(test_case.prompt)
    elapsed = time.time() - start
    return result, elapsed


def run_instruct_test(client: LlamaInstructClient, test_case) -> Tuple[Dict, float]:
    """Run a single test with Llama Instruct."""
    start = time.time()
    result = client.classify_as_guard(test_case.prompt)
    elapsed = time.time() - start
    return result, elapsed


def run_all_tests(verbose: bool = False) -> List[TestResult]:
    """Run all tests on both models."""
    console.print("\n[bold cyan]Initializing clients...[/bold cyan]")

    guard_client = LlamaGuardClient(PASSWORD)
    instruct_client = LlamaInstructClient(PASSWORD)

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task(f"[cyan]Testing {len(TEST_CASES)} cases...", total=len(TEST_CASES))

        for i, test_case in enumerate(TEST_CASES, 1):
            result = TestResult(
                test_case.id,
                test_case.prompt,
                test_case.expected_safe,
                test_case.expected_categories
            )

            if verbose:
                console.print(f"\n[yellow]Test {i}/{len(TEST_CASES)}: {test_case.id}[/yellow]")
                console.print(f"Prompt: {test_case.prompt[:80]}...")

            # Test with Llama Guard
            try:
                guard_result, guard_time = run_guard_test(guard_client, test_case)
                result.guard_safe = guard_result["safe"]
                result.guard_categories = guard_result["categories"]
                result.guard_raw = guard_result["raw_text"]
                result.guard_error = guard_result.get("error")
                result.guard_time = guard_time

                if verbose:
                    console.print(f"  Guard: {'safe' if result.guard_safe else 'unsafe'} {result.guard_categories}")
            except Exception as e:
                result.guard_error = str(e)
                if verbose:
                    console.print(f"  [red]Guard error: {e}[/red]")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

            # Test with Llama Instruct
            try:
                instruct_result, instruct_time = run_instruct_test(instruct_client, test_case)
                result.instruct_safe = instruct_result["safe"]
                result.instruct_categories = instruct_result["categories"]
                result.instruct_raw = instruct_result["raw_text"]
                result.instruct_error = instruct_result.get("error")
                result.instruct_time = instruct_time

                if verbose:
                    console.print(f"  Instruct: {'safe' if result.instruct_safe else 'unsafe'} {result.instruct_categories}")
            except Exception as e:
                result.instruct_error = str(e)
                if verbose:
                    console.print(f"  [red]Instruct error: {e}[/red]")

            results.append(result)
            progress.update(task, advance=1)

            # Small delay between test cases
            time.sleep(0.5)

    return results


def calculate_metrics(results: List[TestResult], model_type: str) -> Dict:
    """Calculate performance metrics for a model."""

    # Extract predictions and ground truth
    y_true = []
    y_pred = []
    valid_results = []

    for result in results:
        expected = result.expected_safe

        if model_type == "guard":
            predicted = result.guard_safe
            error = result.guard_error
        else:  # instruct
            predicted = result.instruct_safe
            error = result.instruct_error

        # Skip if error or None
        if predicted is None or error:
            continue

        y_true.append(1 if expected else 0)  # 1 = safe, 0 = unsafe
        y_pred.append(1 if predicted else 0)
        valid_results.append(result)

    if not y_true:
        return {
            "error": "No valid predictions",
            "total_tests": len(results),
            "valid_tests": 0
        }

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate rates
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    # Average response time
    if model_type == "guard":
        times = [r.guard_time for r in valid_results]
    else:
        times = [r.instruct_time for r in valid_results]

    avg_time = sum(times) / len(times) if times else 0

    return {
        "total_tests": len(results),
        "valid_tests": len(valid_results),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "avg_response_time": avg_time,
        "error": None
    }


def print_comparison_report(results: List[TestResult]):
    """Print a detailed comparison report."""

    console.print("\n[bold cyan]═══════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]     SHIELD COMPARISON TEST REPORT[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════[/bold cyan]\n")

    # Calculate metrics for both models
    guard_metrics = calculate_metrics(results, "guard")
    instruct_metrics = calculate_metrics(results, "instruct")

    # Overall Metrics Table
    table = Table(title="Overall Performance Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Llama Guard 3 8B", justify="right", style="green")
    table.add_column("Llama 3 8B Instruct", justify="right", style="yellow")
    table.add_column("Difference", justify="right")

    metrics_to_show = [
        ("Total Tests", "total_tests", "{:.0f}"),
        ("Valid Tests", "valid_tests", "{:.0f}"),
        ("Accuracy", "accuracy", "{:.2%}"),
        ("Precision", "precision", "{:.2%}"),
        ("Recall", "recall", "{:.2%}"),
        ("F1 Score", "f1_score", "{:.3f}"),
        ("False Positive Rate", "false_positive_rate", "{:.2%}"),
        ("False Negative Rate", "false_negative_rate", "{:.2%}"),
        ("Avg Response Time (s)", "avg_response_time", "{:.3f}"),
    ]

    for label, key, fmt in metrics_to_show:
        if key in guard_metrics and key in instruct_metrics:
            guard_val = guard_metrics[key]
            instruct_val = instruct_metrics[key]

            if isinstance(guard_val, (int, float)) and isinstance(instruct_val, (int, float)):
                diff = guard_val - instruct_val
                diff_str = fmt.format(diff)

                # Color code the difference
                if key in ["accuracy", "precision", "recall", "f1_score"]:
                    # Higher is better
                    diff_color = "green" if diff > 0 else "red" if diff < 0 else "white"
                elif key in ["false_positive_rate", "false_negative_rate", "avg_response_time"]:
                    # Lower is better
                    diff_color = "red" if diff > 0 else "green" if diff < 0 else "white"
                else:
                    diff_color = "white"

                table.add_row(
                    label,
                    fmt.format(guard_val),
                    fmt.format(instruct_val),
                    f"[{diff_color}]{diff_str}[/{diff_color}]"
                )
            else:
                table.add_row(label, str(guard_val), str(instruct_val), "-")

    console.print(table)

    # Confusion Matrix
    console.print("\n[bold]Confusion Matrix Details:[/bold]\n")

    cm_table = Table(show_header=True, header_style="bold")
    cm_table.add_column("Model")
    cm_table.add_column("True Positives", justify="right")
    cm_table.add_column("True Negatives", justify="right")
    cm_table.add_column("False Positives", justify="right")
    cm_table.add_column("False Negatives", justify="right")

    cm_table.add_row(
        "Llama Guard",
        str(guard_metrics.get("true_positives", 0)),
        str(guard_metrics.get("true_negatives", 0)),
        f"[red]{guard_metrics.get('false_positives', 0)}[/red]",
        f"[red]{guard_metrics.get('false_negatives', 0)}[/red]"
    )

    cm_table.add_row(
        "Llama Instruct",
        str(instruct_metrics.get("true_positives", 0)),
        str(instruct_metrics.get("true_negatives", 0)),
        f"[red]{instruct_metrics.get('false_positives', 0)}[/red]",
        f"[red]{instruct_metrics.get('false_negatives', 0)}[/red]"
    )

    console.print(cm_table)

    # Show some example disagreements
    console.print("\n[bold]Example Disagreements:[/bold]\n")

    disagreements = []
    for result in results:
        if result.guard_safe is not None and result.instruct_safe is not None:
            if result.guard_safe != result.instruct_safe:
                disagreements.append(result)

    if disagreements:
        for i, result in enumerate(disagreements[:5], 1):  # Show first 5
            console.print(f"[yellow]{i}. {result.test_id}[/yellow]")
            console.print(f"   Prompt: {result.prompt[:100]}...")
            console.print(f"   Expected: {'safe' if result.expected_safe else 'unsafe'}")
            console.print(f"   Guard: {'safe' if result.guard_safe else 'unsafe'} {result.guard_categories}")
            console.print(f"   Instruct: {'safe' if result.instruct_safe else 'unsafe'} {result.instruct_categories}")
            console.print()
    else:
        console.print("[green]No disagreements found![/green]")

    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════════════[/bold cyan]")
    console.print("[bold]Key Findings:[/bold]")

    guard_f1 = guard_metrics.get("f1_score", 0)
    instruct_f1 = instruct_metrics.get("f1_score", 0)

    if guard_f1 > instruct_f1:
        diff_pct = (guard_f1 - instruct_f1) / instruct_f1 * 100 if instruct_f1 > 0 else 0
        console.print(f"• Llama Guard outperforms Llama Instruct by [green]{diff_pct:.1f}%[/green] (F1 score)")
    elif instruct_f1 > guard_f1:
        diff_pct = (instruct_f1 - guard_f1) / guard_f1 * 100 if guard_f1 > 0 else 0
        console.print(f"• Llama Instruct outperforms Llama Guard by [yellow]{diff_pct:.1f}%[/yellow] (F1 score)")
    else:
        console.print("• Both models perform similarly")

    console.print(f"• Total disagreements: {len(disagreements)} / {len(results)} ({len(disagreements)/len(results)*100:.1f}%)")

    console.print("[bold cyan]═══════════════════════════════════════════════[/bold cyan]\n")


def main():
    parser = argparse.ArgumentParser(description="Compare Llama Guard vs Llama Instruct for safety classification")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output during testing")
    parser.add_argument("--report", "-r", action="store_true", help="Generate detailed report")
    parser.add_argument("--output", "-o", type=str, help="Output file for JSON results")

    args = parser.parse_args()

    console.print("[bold green]Starting Shield Comparison Tests...[/bold green]")

    # Run tests
    results = run_all_tests(verbose=args.verbose)

    # Print report
    print_comparison_report(results)

    # Save detailed results if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "results": [r.to_dict() for r in results],
            "metrics": {
                "guard": calculate_metrics(results, "guard"),
                "instruct": calculate_metrics(results, "instruct")
            }
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"\n[green]✓ Detailed results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
