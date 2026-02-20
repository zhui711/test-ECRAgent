import json
import argparse
import math
from pathlib import Path
import sys

def analyze_cost(input_file):
    print(f"Loading Cost Audit Data from: {input_file}")
    
    latencies_total = []
    latencies_phase1 = []
    latencies_net = []     # Total - Phase1
    latencies_search = []
    latencies_compute = [] # Total - Search (or from JSON)
    
    counts_llm = []
    counts_api = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except:
                continue
                
            costs = data.get("cost_metrics", {})
            if not costs:
                continue
            
            # Extract Metrics
            tot = costs.get("total_latency", 0.0)
            p1 = costs.get("phase1_latency", 0.0)
            search = costs.get("search_latency", 0.0)
            comp = costs.get("compute_latency", 0.0)
            llm = costs.get("llm_calls", 0)
            api = costs.get("api_calls", 0)
            
            # Derived Metrics
            net = tot - p1
            
            latencies_total.append(tot)
            latencies_phase1.append(p1)
            latencies_net.append(net)
            latencies_search.append(search)
            latencies_compute.append(comp)
            
            counts_llm.append(llm)
            counts_api.append(api)
            
    n = len(latencies_total)
    print(f"Analyzed {n} samples.")
    
    if n == 0:
        return

    def print_stat(name, data, unit="s"):
        if not data:
            print(f"{name:<20} | Mean: {0.0:8.2f} {unit} | Std: {0.0:8.2f} {unit}")
            return
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = math.sqrt(variance)
        print(f"{name:<20} | Mean: {mean:8.2f} {unit} | Std: {std:8.2f} {unit}")

    print("-" * 60)
    print("Cost Audit Report (N={})".format(n))
    print("-" * 60)
    
    print_stat("Total Latency", latencies_total)
    print_stat("Phase 1 Latency", latencies_phase1)
    print_stat("Net Latency", latencies_net)
    print("-" * 60)
    print_stat("Search Latency", latencies_search)
    print_stat("Compute Latency", latencies_compute)
    print("-" * 60)
    print_stat("LLM Calls", counts_llm, unit="")
    print_stat("API Calls", counts_api, unit="")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to results_detail.jsonl")
    args = parser.parse_args()
    
    analyze_cost(args.input)
