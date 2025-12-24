#!/usr/bin/env python3
"""
Combine results from multiple model runs into one file.

Usage:
    python combine_results.py results_*.json -o combined_results.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Combine multiple result files")
    parser.add_argument("files", nargs="+", help="Result files to combine")
    parser.add_argument("-o", "--output", default="combined_results.json", help="Output file")
    
    args = parser.parse_args()
    
    all_results = []
    stats = defaultdict(int)
    
    for filepath in args.files:
        print(f"Loading {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for item in data:
            all_results.append(item)
            stats[item.get("model", "unknown")] += 1
    
    # Save combined
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"COMBINED {len(all_results)} results")
    print(f"{'='*50}")
    for model, count in sorted(stats.items()):
        print(f"  {model}: {count}")
    print(f"{'='*50}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
