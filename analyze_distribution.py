"""
Analyze the class distribution of the CheXpert dataset.

This script provides comprehensive statistics about disease prevalence,
class imbalance, and co-occurrence patterns in the dataset.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CHEXPERT_CLASSES


def analyze_class_distribution(csv_path: str) -> Tuple[Dict[str, float], Dict[str, int], int]:
    """
    Analyze the class distribution in the CheXpert dataset.
    
    Args:
        csv_path: Path to the training CSV file
        
    Returns:
        Tuple of (ratios_dict, counts_dict, total_samples)
    """
    df = pd.read_csv(csv_path)
    
    # Fill NaN with 0.0 (negative)
    df[CHEXPERT_CLASSES] = df[CHEXPERT_CLASSES].fillna(0.0)
    
    # Calculate statistics for each disease
    class_distribution_ratio = {}
    class_distribution_count = {}
    total_samples = len(df)
    
    for disease in CHEXPERT_CLASSES:
        positive_samples = (df[disease] == 1.0).sum()
        uncertain_samples = (df[disease] == -1.0).sum()
        
        class_distribution_ratio[disease] = positive_samples / total_samples
        class_distribution_count[disease] = {
            'positive': int(positive_samples),
            'uncertain': int(uncertain_samples),
            'negative': int(total_samples - positive_samples - uncertain_samples)
        }
        
    return class_distribution_ratio, class_distribution_count, total_samples


def print_basic_distribution(
    ratios: Dict[str, float],
    counts: Dict[str, Dict[str, int]],
    total: int
):
    """Print basic class distribution in a readable format."""
    print("\n" + "="*90)
    print("CHEXPERT TRAINING DATASET DISTRIBUTION ANALYSIS")
    print("="*90)
    print(f"\nTotal Samples: {total:,}\n")
    
    # Sort by prevalence
    sorted_dist = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Disease':<35} | {'Positive':>8} | {'Uncertain':>10} | {'Negative':>8} | {'Ratio':>7}")
    print("-" * 90)
    
    for disease, ratio in sorted_dist:
        count_dict = counts[disease]
        positive = count_dict['positive']
        uncertain = count_dict['uncertain']
        negative = count_dict['negative']
        
        print(f"{disease:<35} | {positive:>8} | {uncertain:>10} | {negative:>8} | {ratio*100:>6.2f}%")
    
    print("=" * 90 + "\n")


def print_visual_distribution(
    ratios: Dict[str, float],
    total: int
):
    """Print visual distribution with bar charts."""
    print("\nVISUAL DISTRIBUTION (by positive ratio)")
    print("="*90)
    
    sorted_dist = sorted(ratios.items(), key=lambda x: x[1])
    
    for disease, ratio in sorted_dist:
        count = int(ratio * total)
        bar_length = int(ratio * 60)
        bar = "█" * bar_length + "░" * (60 - bar_length)
        try:
            print(f"{disease:35} | {bar} | {count:5d} ({ratio*100:5.2f}%)")
        except UnicodeEncodeError:
            # Fallback for terminals that don't support block characters (e.g., Windows cp1252)
            ascii_bar = "#" * bar_length + "-" * (60 - bar_length)
            print(f"{disease:35} | {ascii_bar} | {count:5d} ({ratio*100:5.2f}%)")
    
    print("=" * 90 + "\n")


def analyze_co_occurrence(csv_path: str) -> Dict[str, Dict[str, int]]:
    """
    Analyze disease co-occurrence patterns.
    
    Returns a dictionary showing how often diseases appear together.
    """
    df = pd.read_csv(csv_path)
    df[CHEXPERT_CLASSES] = df[CHEXPERT_CLASSES].fillna(0.0)
    
    # Only consider positive samples
    positive_df = df[CHEXPERT_CLASSES][(df[CHEXPERT_CLASSES] == 1.0).any(axis=1)]
    
    co_occurrence = {}
    
    for disease in CHEXPERT_CLASSES:
        co_occurrence[disease] = {}
        disease_positive = positive_df[positive_df[disease] == 1.0]
        
        for other_disease in CHEXPERT_CLASSES:
            if disease != other_disease:
                co_count = (disease_positive[other_disease] == 1.0).sum()
                co_occurrence[disease][other_disease] = int(co_count)
    
    return co_occurrence


def print_co_occurrence(co_occurrence: Dict[str, Dict[str, int]]):
    """Print top disease co-occurrences."""
    print("\nTOP DISEASE CO-OCCURRENCES")
    print("="*90)
    
    # Flatten to get top pairs
    pairs = []
    for disease1, co_dict in co_occurrence.items():
        for disease2, count in co_dict.items():
            if disease1 < disease2:  # Avoid duplicates
                pairs.append((disease1, disease2, count))
    
    # Sort by count
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Disease 1':<30} + {'Disease 2':<30} | {'Count':>6}")
    print("-" * 90)
    
    for disease1, disease2, count in pairs[:15]:  # Top 15
        print(f"{disease1:<30} + {disease2:<30} | {count:>6}")
    
    print("=" * 90 + "\n")


def analyze_metadata_distribution(csv_path: str):
    """Analyze metadata distribution (age, sex, view)."""
    df = pd.read_csv(csv_path)
    
    print("\nMETADATA DISTRIBUTION")
    print("="*90)
    
    # Age statistics
    ages = df['Age'].dropna()
    if len(ages) > 0:
        print(f"\nAge Statistics:")
        print(f"  Mean:   {ages.mean():.2f} years")
        print(f"  Median: {ages.median():.2f} years")
        print(f"  Min:    {ages.min():.2f} years")
        print(f"  Max:    {ages.max():.2f} years")
    
    # Sex distribution
    print(f"\nSex Distribution:")
    sex_dist = df['Sex'].value_counts()
    for sex, count in sex_dist.items():
        pct = count / len(df) * 100
        print(f"  {sex:<10}: {count:>6} ({pct:>5.2f}%)")
    
    # View distribution
    print(f"\nView Distribution (Frontal/Lateral):")
    view_dist = df['Frontal/Lateral'].value_counts()
    for view, count in view_dist.items():
        pct = count / len(df) * 100
        print(f"  {view:<10}: {count:>6} ({pct:>5.2f}%)")
    
    # AP/PA distribution
    print(f"\nView Distribution (AP/PA):")
    ap_pa_dist = df['AP/PA'].value_counts()
    for ap_pa, count in ap_pa_dist.items():
        pct = count / len(df) * 100
        print(f"  {ap_pa:<10}: {count:>6} ({pct:>5.2f}%)")
    
    print("=" * 90 + "\n")


def calculate_balancing_stats(
    ratios: Dict[str, float],
    total: int,
    target_ratio: float = 0.15
) -> Dict[str, int]:
    """Calculate how many samples are needed to balance the dataset."""
    balancing_stats = {}
    
    for disease, ratio in ratios.items():
        current_count = int(ratio * total)
        target_count = int(target_ratio * total)
        needed = max(0, target_count - current_count)
        balancing_stats[disease] = needed
    
    return balancing_stats


def print_balancing_stats(
    ratios: Dict[str, float],
    counts: Dict[str, Dict[str, int]],
    total: int,
    target_ratio: float = 0.15
):
    """Print statistics about dataset balancing."""
    print("\nDATASET BALANCING REQUIREMENTS")
    print("="*90)
    print(f"Target positive ratio: {target_ratio*100:.2f}%\n")
    
    stats = calculate_balancing_stats(ratios, total, target_ratio)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Disease':<35} | {'Current':>8} | {'Target':>8} | {'Needed':>8}")
    print("-" * 90)
    
    total_needed = 0
    for disease, needed in sorted_stats:
        current = counts[disease]['positive']
        target = int(target_ratio * total)
        total_needed += needed
        print(f"{disease:<35} | {current:>8} | {target:>8} | {needed:>8}")
    
    print("-" * 90)
    print(f"{'TOTAL SYNTHETIC IMAGES NEEDED':<35} | {'':>8} | {'':>8} | {total_needed:>8}")
    print("=" * 90 + "\n")


def plot_distribution(
    ratios: Dict[str, float],
    output_path: str = "distribution_analysis.png"
):
    """Create visualization of class distribution."""
    try:
        diseases = list(ratios.keys())
        ratios_list = list(ratios.values())
        
        # Sort by ratio
        sorted_pairs = sorted(zip(diseases, ratios_list), key=lambda x: x[1])
        diseases_sorted = [p[0] for p in sorted_pairs]
        ratios_sorted = [p[1] for p in sorted_pairs]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#ff7f0e' if r < 0.1 else '#2ca02c' for r in ratios_sorted]
        bars = ax.barh(diseases_sorted, ratios_sorted, color=colors, alpha=0.7)
        
        ax.set_xlabel('Positive Ratio (%)', fontsize=12)
        ax.set_title('CheXpert Training Dataset - Disease Distribution', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(ratios_sorted) * 1.1)
        
        # Add percentage labels
        for i, (bar, ratio) in enumerate(zip(bars, ratios_sorted)):
            ax.text(ratio + 0.005, i, f'{ratio*100:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {output_path}")
        
    except Exception as e:
        print(f"Warning: Could not create plot - {e}")


def plot_classwise_distribution(
    counts: Dict[str, Dict[str, int]],
    output_path: str = "distribution_by_class.png"
):
    """Create a stacked bar chart with positive/uncertain/negative counts per class."""
    try:
        diseases = list(counts.keys())
        positives = [counts[d]['positive'] for d in diseases]
        uncertain = [counts[d]['uncertain'] for d in diseases]
        negative = [counts[d]['negative'] for d in diseases]

        ind = list(range(len(diseases)))
        fig, ax = plt.subplots(figsize=(14, 8))

        p1 = ax.bar(ind, positives, label='Positive', color='#d62728')
        p2 = ax.bar(ind, uncertain, bottom=positives, label='Uncertain', color='#ff7f0e')
        bottoms = [p + u for p, u in zip(positives, uncertain)]
        p3 = ax.bar(ind, negative, bottom=bottoms, label='Negative', color='#1f77b4')

        ax.set_xticks(ind)
        ax.set_xticklabels(diseases, rotation=90)
        ax.set_ylabel('Count')
        ax.set_title('Per-class counts (Positive / Uncertain / Negative)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Per-class distribution plot saved to {output_path}")

    except Exception as e:
        print(f"Warning: Could not create per-class plot - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CheXpert dataset class distribution"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default="CheXpert-v1.0-small/train.csv",
        help="Path to training CSV file"
    )
    
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.15,
        help="Target positive ratio for balancing analysis"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization of distribution"
    )
    
    parser.add_argument(
        "--output-chart",
        type=str,
        default="distribution_analysis.png",
        help="Path to save distribution chart"
    )
    
    parser.add_argument(
        "--co-occurrence",
        action="store_true",
        help="Show disease co-occurrence patterns"
    )
    
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Show metadata distribution (age, sex, view)"
    )
    
    args = parser.parse_args()
    
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Analyze distribution
    print(f"\nAnalyzing {args.csv}...")
    ratios, counts, total = analyze_class_distribution(args.csv)
    
    # Print results
    print_basic_distribution(ratios, counts, total)
    print_visual_distribution(ratios, total)
    print_balancing_stats(ratios, counts, total, args.target_ratio)
    
    # Optional: co-occurrence analysis
    if args.co_occurrence:
        print("Analyzing disease co-occurrence...")
        co_occurrence = analyze_co_occurrence(args.csv)
        print_co_occurrence(co_occurrence)
    
    # Optional: metadata analysis
    if args.metadata:
        analyze_metadata_distribution(args.csv)
    
    # Optional: create plot
    if args.plot:
        plot_distribution(ratios, args.output_chart)
        # also create a per-class stacked count plot
        perclass_path = args.output_chart.replace('.png', '') + '_per_class.png'
        plot_classwise_distribution(counts, perclass_path)


if __name__ == "__main__":
    main()
