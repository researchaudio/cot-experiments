#!/usr/bin/env python3
"""
Analyze COT faithfulness results - works with single or combined model data.

Usage:
    python analyze.py results.json
    python analyze.py combined_all_models.json
"""

import json
import argparse
from collections import defaultdict
from datetime import datetime

def analyze(filepath: str):
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"COT FAITHFULNESS ANALYSIS")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    print(f"Total trials: {len(results)}")
    
    # Get time range
    times = [r.get("timestamp", "") for r in results if r.get("timestamp")]
    if times:
        print(f"Time range: {min(times)[:19]} to {max(times)[:19]}")
    
    models = sorted(set(r.get("model", "unknown") for r in results))
    print(f"Models: {len(models)}")
    for m in models:
        count = len([r for r in results if r.get("model") == m])
        print(f"  - {m}: {count} trials")
    
    # ========================================
    # 1. OVERALL METRICS
    # ========================================
    print(f"\n{'='*70}")
    print("1. OVERALL METRICS")
    print(f"{'='*70}")
    
    faithful = sum(1 for r in results if r.get("is_faithful"))
    hidden = sum(1 for r in results if r.get("hidden_knowledge"))
    used_hint = sum(1 for r in results if r.get("used_hint"))
    acknowledged = sum(1 for r in results if r.get("acknowledged_in_response"))
    
    print(f"  Faithful:           {faithful:4}/{len(results)} ({faithful/len(results)*100:5.1f}%)")
    print(f"  Hidden knowledge:   {hidden:4}/{len(results)} ({hidden/len(results)*100:5.1f}%)")
    print(f"  Used hint:          {used_hint:4}/{len(results)} ({used_hint/len(results)*100:5.1f}%)")
    print(f"  Acknowledged hint:  {acknowledged:4}/{len(results)} ({acknowledged/len(results)*100:5.1f}%)")
    
    # ========================================
    # 2. BY MODEL
    # ========================================
    print(f"\n{'='*70}")
    print("2. FAITHFULNESS BY MODEL")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Faithful':>10} {'Hidden':>10} {'Used Hint':>10}")
    print("-" * 60)
    
    model_stats = {}
    for model in models:
        mr = [r for r in results if r.get("model") == model]
        f_count = sum(1 for r in mr if r.get("is_faithful"))
        h_count = sum(1 for r in mr if r.get("hidden_knowledge"))
        u_count = sum(1 for r in mr if r.get("used_hint"))
        
        f_rate = f_count / len(mr) * 100 if mr else 0
        h_rate = h_count / len(mr) * 100 if mr else 0
        u_rate = u_count / len(mr) * 100 if mr else 0
        
        model_stats[model] = {"faithful": f_rate, "hidden": h_rate, "used_hint": u_rate, "n": len(mr)}
        print(f"{model:<25} {f_rate:>9.1f}% {h_rate:>9.1f}% {u_rate:>9.1f}%  (n={len(mr)})")
    
    # ========================================
    # 3. BY SYSTEM TYPE
    # ========================================
    print(f"\n{'='*70}")
    print("3. FAITHFULNESS BY SYSTEM TYPE (ALL MODELS)")
    print(f"{'='*70}")
    
    sys_types = ["baseline", "monitoring", "explicit_ack"]
    sys_stats = {}
    
    print(f"{'System Type':<15} {'Faithful':>10} {'Hidden':>10} {'N':>8}")
    print("-" * 50)
    
    for st in sys_types:
        sr = [r for r in results if r.get("system_type") == st]
        if sr:
            f_count = sum(1 for r in sr if r.get("is_faithful"))
            h_count = sum(1 for r in sr if r.get("hidden_knowledge"))
            f_rate = f_count / len(sr) * 100
            h_rate = h_count / len(sr) * 100
            sys_stats[st] = {"faithful": f_rate, "hidden": h_rate, "n": len(sr)}
            print(f"{st:<15} {f_rate:>9.1f}% {h_rate:>9.1f}% {len(sr):>8}")
    
    # ========================================
    # 4. BY MODEL × SYSTEM TYPE (THE KEY TABLE)
    # ========================================
    print(f"\n{'='*70}")
    print("4. FAITHFULNESS BY MODEL × SYSTEM TYPE")
    print(f"{'='*70}")
    
    # Header
    print(f"{'Model':<25}", end="")
    for st in sys_types:
        print(f" {st:>12}", end="")
    print(f" {'Δ(exp-base)':>12}")
    print("-" * 75)
    
    for model in models:
        print(f"{model:<25}", end="")
        rates = {}
        for st in sys_types:
            mr = [r for r in results if r.get("model") == model and r.get("system_type") == st]
            if mr:
                f_rate = sum(1 for r in mr if r.get("is_faithful")) / len(mr) * 100
                rates[st] = f_rate
                print(f" {f_rate:>11.1f}%", end="")
            else:
                print(f" {'N/A':>12}", end="")
        
        # Delta
        if "baseline" in rates and "explicit_ack" in rates:
            delta = rates["explicit_ack"] - rates["baseline"]
            print(f" {delta:>+11.1f}pp")
        else:
            print(f" {'N/A':>12}")
    
    # ========================================
    # 5. CATEGORY BREAKDOWN
    # ========================================
    print(f"\n{'='*70}")
    print("5. CATEGORY BREAKDOWN")
    print(f"{'='*70}")
    
    categories = defaultdict(int)
    for r in results:
        categories[r.get("category", "unknown")] += 1
    
    print(f"{'Category':<25} {'Count':>8} {'Percent':>10}")
    print("-" * 45)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"{cat:<25} {count:>8} {count/len(results)*100:>9.1f}%")
    
    # ========================================
    # 6. HIDDEN KNOWLEDGE BY MODEL
    # ========================================
    print(f"\n{'='*70}")
    print("6. HIDDEN KNOWLEDGE DETECTION BY MODEL")
    print(f"{'='*70}")
    
    print(f"{'Model':<25} {'Hidden Cases':>15} {'Rate':>10}")
    print("-" * 55)
    
    for model in models:
        mr = [r for r in results if r.get("model") == model]
        h_count = sum(1 for r in mr if r.get("hidden_knowledge"))
        h_rate = h_count / len(mr) * 100 if mr else 0
        print(f"{model:<25} {h_count:>15} {h_rate:>9.1f}%")
    
    # ========================================
    # 7. KEY FINDINGS SUMMARY
    # ========================================
    print(f"\n{'='*70}")
    print("7. KEY FINDINGS")
    print(f"{'='*70}")
    
    # Best/worst baseline
    if model_stats:
        baseline_rates = {}
        explicit_rates = {}
        for model in models:
            br = [r for r in results if r.get("model") == model and r.get("system_type") == "baseline"]
            er = [r for r in results if r.get("model") == model and r.get("system_type") == "explicit_ack"]
            if br:
                baseline_rates[model] = sum(1 for r in br if r.get("is_faithful")) / len(br) * 100
            if er:
                explicit_rates[model] = sum(1 for r in er if r.get("is_faithful")) / len(er) * 100
        
        if baseline_rates:
            best_base = max(baseline_rates, key=baseline_rates.get)
            worst_base = min(baseline_rates, key=baseline_rates.get)
            print(f"  Highest baseline faithfulness: {best_base} ({baseline_rates[best_base]:.1f}%)")
            print(f"  Lowest baseline faithfulness:  {worst_base} ({baseline_rates[worst_base]:.1f}%)")
        
        if baseline_rates and explicit_rates:
            # Biggest improvement
            improvements = {m: explicit_rates.get(m, 0) - baseline_rates.get(m, 0) 
                          for m in models if m in baseline_rates and m in explicit_rates}
            if improvements:
                best_improve = max(improvements, key=improvements.get)
                print(f"  Biggest improvement with explicit_ack: {best_improve} (+{improvements[best_improve]:.1f}pp)")
    
    # Overall effect
    if "baseline" in sys_stats and "explicit_ack" in sys_stats:
        base_f = sys_stats["baseline"]["faithful"]
        exp_f = sys_stats["explicit_ack"]["faithful"]
        print(f"\n  🔑 EXPLICIT ACKNOWLEDGMENT EFFECT: {base_f:.1f}% → {exp_f:.1f}% (+{exp_f-base_f:.1f}pp)")
    
    if "baseline" in sys_stats and "monitoring" in sys_stats:
        base_f = sys_stats["baseline"]["faithful"]
        mon_f = sys_stats["monitoring"]["faithful"]
        print(f"  🔑 MONITORING EFFECT: {base_f:.1f}% → {mon_f:.1f}% ({mon_f-base_f:+.1f}pp)")
    
    # Hidden knowledge
    total_hidden = sum(1 for r in results if r.get("hidden_knowledge"))
    print(f"\n  ⚠️  HIDDEN KNOWLEDGE: {total_hidden} cases ({total_hidden/len(results)*100:.1f}%) detected via followup probes")
    
    print(f"\n{'='*70}\n")
    
    # ========================================
    # 8. SAVE SUMMARY TO FILE
    # ========================================
    import os
    basename = os.path.basename(filepath).replace(".json", "_analysis.txt")
    summary_file = basename  # Save in current directory
    try:
        with open(summary_file, 'w') as f:
            f.write(f"COT Faithfulness Analysis Summary\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Source: {filepath}\n")
            f.write(f"Total trials: {len(results)}\n")
            f.write(f"Models: {', '.join(models)}\n\n")
            
            f.write("Faithfulness by System Type:\n")
            for st, stats in sys_stats.items():
                f.write(f"  {st}: {stats['faithful']:.1f}%\n")
            
            f.write(f"\nHidden knowledge rate: {total_hidden/len(results)*100:.1f}%\n")
        print(f"📄 Summary saved to: {summary_file}")
    except Exception as e:
        print(f"(Could not save summary file: {e})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Results JSON file")
    args = parser.parse_args()
    analyze(args.file)
