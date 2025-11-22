#!/usr/bin/env python3
"""
Summary Comparison Tool

Analyzes and compares the different summary types generated for each collection,
helping you understand the trade-offs between different processing approaches.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

def parse_summary_file(filepath):
    """
    Parse a summary file and extract metadata, usage stats, and content.
    
    Args:
        filepath: Path to summary file
    
    Returns:
        dict with 'metadata', 'usage', and 'summary' keys
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None
    
    result = {
        'metadata': {},
        'usage': {},
        'summary': ''
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    section = None
    summary_lines = []
    
    for line in lines:
        if 'METADATA:' in line:
            section = 'metadata'
            continue
        elif 'TOKEN USAGE STATISTICS:' in line:
            section = 'usage'
            continue
        elif 'GAMEPLAY SUMMARY:' in line:
            section = 'summary'
            continue
        elif section == 'summary' and '=' * 40 in line:
            # End of summary section
            section = None
            continue
        
        if section == 'metadata' and ':' in line and not line.startswith('-'):
            key, value = line.split(':', 1)
            result['metadata'][key.strip()] = value.strip()
        
        elif section == 'usage' and ':' in line and not line.startswith('-'):
            key, value = line.split(':', 1)
            value = value.strip().replace(',', '')
            try:
                result['usage'][key.strip()] = int(value)
            except ValueError:
                result['usage'][key.strip()] = value
        
        elif section == 'summary':
            summary_lines.append(line)
    
    result['summary'] = '\n'.join(summary_lines).strip()
    return result

def analyze_collection(collection_path):
    """
    Analyze all summaries in a collection directory.
    
    Args:
        collection_path: Path to collection directory
    
    Returns:
        dict with analysis results
    """
    collection_path = Path(collection_path)
    
    summary_types = [
        'raw_video_no_log',
        'keyframes_log_unparsed',
        'keyframes_log_parsed',
        'event_frames_log'
    ]
    
    results = {}
    
    for summary_type in summary_types:
        filepath = collection_path / f"{summary_type}.txt"
        parsed = parse_summary_file(filepath)
        
        if parsed:
            results[summary_type] = {
                'exists': True,
                'data': parsed,
                'file': str(filepath)
            }
        else:
            results[summary_type] = {
                'exists': False,
                'file': str(filepath)
            }
    
    return results

def print_collection_comparison(collection_name, analysis):
    """
    Print a comparison of all summaries for a collection.
    
    Args:
        collection_name: Name of the collection
        analysis: Analysis results from analyze_collection
    """
    print("\n" + "=" * 80)
    print(f"Collection: {collection_name}")
    print("=" * 80)
    
    # Summary table
    print("\nSummary Availability:")
    print("-" * 80)
    print(f"{'Summary Type':<30} {'Status':<15} {'Tokens':<15} {'Cost (est.)':<15}")
    print("-" * 80)
    
    total_tokens = 0
    
    for summary_type, result in analysis.items():
        if result['exists']:
            usage = result['data']['usage']
            tokens = usage.get('Total tokens', 0)
            total_tokens += tokens
            
            # Rough cost estimate (GPT-5 pricing: ~$0.01 per 1k tokens)
            cost = tokens * 0.00001
            
            status = "✓ Generated"
            tokens_str = f"{tokens:,}"
            cost_str = f"${cost:.2f}"
        else:
            status = "✗ Missing"
            tokens_str = "-"
            cost_str = "-"
        
        # Shorten summary type name for display
        display_name = summary_type.replace('_', ' ').title()
        print(f"{display_name:<30} {status:<15} {tokens_str:<15} {cost_str:<15}")
    
    print("-" * 80)
    if total_tokens > 0:
        total_cost = total_tokens * 0.00001
        print(f"{'TOTAL':<30} {'':<15} {total_tokens:,:<15} ${total_cost:.2f}")
    print()
    
    # Summary length comparison
    print("\nSummary Lengths (characters):")
    print("-" * 80)
    
    for summary_type, result in analysis.items():
        if result['exists']:
            summary = result['data']['summary']
            length = len(summary)
            word_count = len(summary.split())
            
            display_name = summary_type.replace('_', ' ').title()
            print(f"{display_name:<30} {length:>10,} chars  ({word_count:>6,} words)")
    
    print()

def print_detailed_comparison(analysis):
    """
    Print detailed comparison of summary content.
    
    Args:
        analysis: Analysis results from analyze_collection
    """
    print("\nDetailed Summary Previews:")
    print("=" * 80)
    
    for summary_type, result in analysis.items():
        if result['exists']:
            print(f"\n{summary_type.upper().replace('_', ' ')}")
            print("-" * 80)
            
            # Print first 500 characters of summary
            summary = result['data']['summary']
            preview = summary[:500]
            if len(summary) > 500:
                preview += "..."
            
            print(preview)
            print()

def generate_comparison_report(sample_data_path, output_file=None):
    """
    Generate a comprehensive comparison report for all collections.
    
    Args:
        sample_data_path: Path to sample_data directory
        output_file: Optional output file path
    """
    sample_data_path = Path(sample_data_path)
    
    if not sample_data_path.exists():
        print(f"Error: Directory not found: {sample_data_path}")
        return
    
    # Find all collections
    collections = [d for d in sample_data_path.iterdir() if d.is_dir()]
    collections = sorted(collections)
    
    if not collections:
        print(f"Error: No collection directories found in {sample_data_path}")
        return
    
    print("=" * 80)
    print("SUMMARY COMPARISON REPORT")
    print("=" * 80)
    print(f"Sample data: {sample_data_path}")
    print(f"Collections found: {len(collections)}")
    print()
    
    # Analyze each collection
    all_analyses = {}
    
    for collection in collections:
        analysis = analyze_collection(collection)
        all_analyses[collection.name] = analysis
        print_collection_comparison(collection.name, analysis)
    
    # Global statistics
    print("\n" + "=" * 80)
    print("GLOBAL STATISTICS")
    print("=" * 80)
    
    summary_type_stats = defaultdict(lambda: {'count': 0, 'total_tokens': 0, 'total_cost': 0})
    
    for collection_name, analysis in all_analyses.items():
        for summary_type, result in analysis.items():
            if result['exists']:
                summary_type_stats[summary_type]['count'] += 1
                tokens = result['data']['usage'].get('Total tokens', 0)
                summary_type_stats[summary_type]['total_tokens'] += tokens
                summary_type_stats[summary_type]['total_cost'] += tokens * 0.00001
    
    print("\nSummary Type Statistics:")
    print("-" * 80)
    print(f"{'Summary Type':<30} {'Generated':<12} {'Total Tokens':<20} {'Total Cost':<15}")
    print("-" * 80)
    
    grand_total_tokens = 0
    grand_total_cost = 0
    
    for summary_type in ['raw_video_no_log', 'keyframes_log_unparsed', 
                          'keyframes_log_parsed', 'event_frames_log']:
        stats = summary_type_stats[summary_type]
        display_name = summary_type.replace('_', ' ').title()
        
        print(f"{display_name:<30} {stats['count']:<12} {stats['total_tokens']:>20,} ${stats['total_cost']:>14.2f}")
        
        grand_total_tokens += stats['total_tokens']
        grand_total_cost += stats['total_cost']
    
    print("-" * 80)
    print(f"{'GRAND TOTAL':<30} {'':<12} {grand_total_tokens:>20,} ${grand_total_cost:>14.2f}")
    print()
    
    print("\nToken Efficiency Analysis:")
    print("-" * 80)
    
    if summary_type_stats['raw_video_no_log']['count'] > 0:
        baseline_avg = (summary_type_stats['raw_video_no_log']['total_tokens'] / 
                       summary_type_stats['raw_video_no_log']['count'])
        
        print(f"Baseline (raw video): {baseline_avg:,.0f} tokens avg")
        
        for summary_type in ['keyframes_log_unparsed', 'keyframes_log_parsed', 'event_frames_log']:
            if summary_type_stats[summary_type]['count'] > 0:
                avg = (summary_type_stats[summary_type]['total_tokens'] / 
                      summary_type_stats[summary_type]['count'])
                
                reduction = ((baseline_avg - avg) / baseline_avg) * 100
                display_name = summary_type.replace('_', ' ').title()
                
                print(f"{display_name}: {avg:,.0f} tokens avg ({reduction:+.1f}% vs baseline)")
    
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="Compare and analyze generated gameplay summaries"
    )
    
    parser.add_argument(
        "--sample_data",
        default="sample_data",
        help="Path to sample_data directory (default: sample_data)"
    )
    
    parser.add_argument(
        "--collection",
        help="Analyze only a specific collection"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed summary previews"
    )
    
    parser.add_argument(
        "--output",
        help="Save report to file"
    )
    
    args = parser.parse_args()
    
    sample_data_path = Path(args.sample_data)
    
    if args.collection:
        # Analyze single collection
        collection_path = sample_data_path / args.collection
        
        if not collection_path.exists():
            print(f"Error: Collection not found: {collection_path}")
            return
        
        analysis = analyze_collection(collection_path)
        print_collection_comparison(args.collection, analysis)
        
        if args.detailed:
            print_detailed_comparison(analysis)
    else:
        # Analyze all collections
        generate_comparison_report(sample_data_path, args.output)

if __name__ == "__main__":
    main()
