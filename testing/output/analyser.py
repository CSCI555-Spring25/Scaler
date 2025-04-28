import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

def analyze_scaler_data(reactive_file, proactive_file):
    # Check if files exist before loading
    if not os.path.exists(reactive_file):
        raise FileNotFoundError(f"Reactive file not found: {reactive_file}")
    if not os.path.exists(proactive_file):
        raise FileNotFoundError(f"Proactive file not found: {proactive_file}")
        
    # Load the data
    reactive_data = pd.read_csv(reactive_file)
    proactive_data = pd.read_csv(proactive_file)
    
    # Skip first 40 rows to ignore initial stabilization period
    reactive_data = reactive_data.iloc[40:900]
    proactive_data = proactive_data.iloc[40:900]
    
    # Column mappings
    column_mapping = {
        'pods': 'pods',
        'avg_latency': 'latency_avg_ms',
        'p50_latency': 'p50_ms',
        'p75_latency': 'p75_ms',
        'p90_latency': 'p90_ms',
        'p99_latency': 'p99_ms',
        'p99_9_latency': 'p99.9_ms',
        'p99_99_latency': 'p99.99_ms',
        'p99_999_latency': 'p99.999_ms',
        'p100_latency': 'p100_ms',
        'normalized_rps': 'req_sec_avg',  # Use req_sec_avg as a proxy for CPU usage
        'success_rate': 'req_sec_avg',    # No direct success rate, using req_sec_avg as proxy
        'queue_length': 'connections',     # Using connections as proxy for queue length
        'requests': 'total_requests'      # Total requests
    }
    
    # 1. Calculate percentage of more total pods in proactive compared to reactive
    avg_reactive_pods = reactive_data[column_mapping['pods']].mean()
    avg_proactive_pods = proactive_data[column_mapping['pods']].mean()
    pod_increase_percentage = ((avg_proactive_pods - avg_reactive_pods) / avg_reactive_pods) * 100
    
    # 2. Calculate average latency for both approaches
    avg_reactive_latency = reactive_data[column_mapping['avg_latency']].mean()
    avg_proactive_latency = proactive_data[column_mapping['avg_latency']].mean()
    latency_improvement_percentage = ((avg_reactive_latency - avg_proactive_latency) / avg_reactive_latency) * 100
    
    # 3. Calculate average CPU usage (using req_sec_avg as a proxy)
    avg_reactive_cpu = reactive_data[column_mapping['normalized_rps']].mean()
    avg_proactive_cpu = proactive_data[column_mapping['normalized_rps']].mean()
    
    # 4. Additional metrics
    # 4.1 Latency percentiles comparison
    avg_reactive_p50 = reactive_data[column_mapping['p50_latency']].mean()
    avg_proactive_p50 = proactive_data[column_mapping['p50_latency']].mean()
    p50_improvement_percentage = ((avg_reactive_p50 - avg_proactive_p50) / avg_reactive_p50) * 100
    
    avg_reactive_p75 = reactive_data[column_mapping['p75_latency']].mean()
    avg_proactive_p75 = proactive_data[column_mapping['p75_latency']].mean()
    p75_improvement_percentage = ((avg_reactive_p75 - avg_proactive_p75) / avg_reactive_p75) * 100
    
    avg_reactive_p90 = reactive_data[column_mapping['p90_latency']].mean()
    avg_proactive_p90 = proactive_data[column_mapping['p90_latency']].mean()
    p90_improvement_percentage = ((avg_reactive_p90 - avg_proactive_p90) / avg_reactive_p90) * 100
    
    avg_reactive_p99 = reactive_data[column_mapping['p99_latency']].mean()
    avg_proactive_p99 = proactive_data[column_mapping['p99_latency']].mean()
    p99_improvement_percentage = ((avg_reactive_p99 - avg_proactive_p99) / avg_reactive_p99) * 100
    
    avg_reactive_p99_9 = reactive_data[column_mapping['p99_9_latency']].mean()
    avg_proactive_p99_9 = proactive_data[column_mapping['p99_9_latency']].mean()
    p99_9_improvement_percentage = ((avg_reactive_p99_9 - avg_proactive_p99_9) / avg_reactive_p99_9) * 100
    
    avg_reactive_p99_99 = reactive_data[column_mapping['p99_99_latency']].mean()
    avg_proactive_p99_99 = proactive_data[column_mapping['p99_99_latency']].mean()
    p99_99_improvement_percentage = ((avg_reactive_p99_99 - avg_proactive_p99_99) / avg_reactive_p99_99) * 100
    
    avg_reactive_p99_999 = reactive_data[column_mapping['p99_999_latency']].mean()
    avg_proactive_p99_999 = proactive_data[column_mapping['p99_999_latency']].mean()
    p99_999_improvement_percentage = ((avg_reactive_p99_999 - avg_proactive_p99_999) / avg_reactive_p99_999) * 100
    
    avg_reactive_p100 = reactive_data[column_mapping['p100_latency']].mean()
    avg_proactive_p100 = proactive_data[column_mapping['p100_latency']].mean()
    p100_improvement_percentage = ((avg_reactive_p100 - avg_proactive_p100) / avg_reactive_p100) * 100
    
    # 4.2 Success rate comparison (using req_sec_avg as proxy since no direct success rate available)
    avg_reactive_success = reactive_data[column_mapping['success_rate']].mean()
    avg_proactive_success = proactive_data[column_mapping['success_rate']].mean()
    success_improvement_percentage = ((avg_proactive_success - avg_reactive_success) / avg_reactive_success) * 100
    
    # 4.3 Queue length comparison (using connections as proxy)
    avg_reactive_queue = reactive_data[column_mapping['queue_length']].mean()
    avg_proactive_queue = proactive_data[column_mapping['queue_length']].mean()
    queue_reduction_percentage = ((avg_reactive_queue - avg_proactive_queue) / avg_reactive_queue) * 100 if avg_reactive_queue > 0 else 0
    
    # Print results
    print("\n===== Reactive vs Proactive Scaler Analysis =====\n")
    
    results = [
        ["Average Pods", f"{avg_reactive_pods:.2f}", f"{avg_proactive_pods:.2f}", f"{pod_increase_percentage:.2f}% {'more' if pod_increase_percentage > 0 else 'fewer'} pods in proactive"],
        ["Average Latency (ms)", f"{avg_reactive_latency:.2f}", f"{avg_proactive_latency:.2f}", f"{abs(latency_improvement_percentage):.2f}% {'improvement' if latency_improvement_percentage > 0 else 'increase'} in proactive"],
        ["Average RPS", f"{avg_reactive_cpu:.2f}", f"{avg_proactive_cpu:.2f}", f"{abs(avg_proactive_cpu - avg_reactive_cpu):.2f} {'higher' if avg_proactive_cpu > avg_reactive_cpu else 'lower'} in proactive"],
        ["P50 Latency (ms)", f"{avg_reactive_p50:.2f}", f"{avg_proactive_p50:.2f}", f"{abs(p50_improvement_percentage):.2f}% {'improvement' if p50_improvement_percentage > 0 else 'increase'} in proactive"],
        ["P75 Latency (ms)", f"{avg_reactive_p75:.2f}", f"{avg_proactive_p75:.2f}", f"{abs(p75_improvement_percentage):.2f}% {'improvement' if p75_improvement_percentage > 0 else 'increase'} in proactive"],
        ["P90 Latency (ms)", f"{avg_reactive_p90:.2f}", f"{avg_proactive_p90:.2f}", f"{abs(p90_improvement_percentage):.2f}% {'improvement' if p90_improvement_percentage > 0 else 'increase'} in proactive"],
        ["P99 Latency (ms)", f"{avg_reactive_p99:.2f}", f"{avg_proactive_p99:.2f}", f"{abs(p99_improvement_percentage):.2f}% {'improvement' if p99_improvement_percentage > 0 else 'increase'} in proactive"],
        ["P99.9 Latency (ms)", f"{avg_reactive_p99_9:.2f}", f"{avg_proactive_p99_9:.2f}", f"{abs(p99_9_improvement_percentage):.2f}% {'improvement' if p99_9_improvement_percentage > 0 else 'increase'} in proactive"],
        ["P99.99 Latency (ms)", f"{avg_reactive_p99_99:.2f}", f"{avg_proactive_p99_99:.2f}", f"{abs(p99_99_improvement_percentage):.2f}% {'improvement' if p99_99_improvement_percentage > 0 else 'increase'} in proactive"],
        ["P99.999 Latency (ms)", f"{avg_reactive_p99_999:.2f}", f"{avg_proactive_p99_999:.2f}", f"{abs(p99_999_improvement_percentage):.2f}% {'improvement' if p99_999_improvement_percentage > 0 else 'increase'} in proactive"],
        ["P100 Latency (ms)", f"{avg_reactive_p100:.2f}", f"{avg_proactive_p100:.2f}", f"{abs(p100_improvement_percentage):.2f}% {'improvement' if p100_improvement_percentage > 0 else 'increase'} in proactive"],
        ["Throughput (RPS)", f"{avg_reactive_success:.2f}", f"{avg_proactive_success:.2f}", f"{abs(success_improvement_percentage):.2f}% {'improvement' if success_improvement_percentage > 0 else 'decrease'} in proactive"],
        ["Average Connections", f"{avg_reactive_queue:.2f}", f"{avg_proactive_queue:.2f}", f"{abs(queue_reduction_percentage):.2f}% {'reduction' if queue_reduction_percentage > 0 else 'increase'} in proactive"]
    ]
    
    print(tabulate(results, headers=["Metric", "Reactive", "Proactive", "Difference"], tablefmt="grid"))
    
    # Create visualizations
    create_comparison_charts(reactive_data, proactive_data, column_mapping)
    
    return {
        "pod_increase_percentage": pod_increase_percentage,
        "avg_reactive_latency": avg_reactive_latency,
        "avg_proactive_latency": avg_proactive_latency,
        "avg_reactive_cpu": avg_reactive_cpu,
        "avg_proactive_cpu": avg_proactive_cpu,
        "p50_improvement_percentage": p50_improvement_percentage,
        "p75_improvement_percentage": p75_improvement_percentage,
        "p90_improvement_percentage": p90_improvement_percentage,
        "p99_improvement_percentage": p99_improvement_percentage,
        "p99_9_improvement_percentage": p99_9_improvement_percentage,
        "p99_99_improvement_percentage": p99_99_improvement_percentage,
        "p99_999_improvement_percentage": p99_999_improvement_percentage,
        "p100_improvement_percentage": p100_improvement_percentage,
        "success_improvement_percentage": success_improvement_percentage
    }

def create_comparison_charts(reactive_data, proactive_data, column_mapping):
    # Time series of pods
    plt.figure(figsize=(12, 6))
    plt.plot(reactive_data.index, reactive_data[column_mapping['pods']], label='Reactive Scaling')
    plt.plot(proactive_data.index, proactive_data[column_mapping['pods']], label='Proactive Scaling')
    plt.title('Number of Pods Over Time')
    plt.xlabel('Time Intervals')
    plt.ylabel('Number of Pods')
    plt.legend()
    plt.grid(True)
    plt.savefig('pods_comparison.png')
    
    # Time series of latency
    plt.figure(figsize=(12, 6))
    plt.plot(reactive_data.index, reactive_data[column_mapping['avg_latency']], label='Reactive Scaling')
    plt.plot(proactive_data.index, proactive_data[column_mapping['avg_latency']], label='Proactive Scaling')
    plt.title('Average Latency Over Time')
    plt.xlabel('Time Intervals')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('latency_comparison.png')
    # Pods vs Requests for Reactive
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time Intervals')
    ax1.set_ylabel('Number of Pods', color=color)
    ax1.plot(reactive_data.index, reactive_data[column_mapping['pods']], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'black'
    ax2.set_ylabel('Total Requests', color=color)
    ax2.plot(reactive_data.index, reactive_data[column_mapping['requests']], color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Reactive: Pods vs Requests Over Time')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig('reactive_pods_requests.png')
    
    # Pods vs Requests for Proactive
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time Intervals')
    ax1.set_ylabel('Number of Pods', color=color)
    ax1.plot(proactive_data.index, proactive_data[column_mapping['pods']], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'black'
    ax2.set_ylabel('Total Requests', color=color)
    ax2.plot(proactive_data.index, proactive_data[column_mapping['requests']], color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Proactive: Pods vs Requests Over Time')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig('proactive_pods_requests.png')
    
    # Bar chart for key metrics
    metrics = ['Average Pods', 'Average Latency (ms)', 'P50 Latency (ms)', 'P90 Latency (ms)', 'P99 Latency (ms)']
    reactive_values = [
        reactive_data[column_mapping['pods']].mean(), 
        reactive_data[column_mapping['avg_latency']].mean(), 
        reactive_data[column_mapping['p50_latency']].mean(),
        reactive_data[column_mapping['p90_latency']].mean(),
        reactive_data[column_mapping['p99_latency']].mean()
    ]
    proactive_values = [
        proactive_data[column_mapping['pods']].mean(), 
        proactive_data[column_mapping['avg_latency']].mean(), 
        proactive_data[column_mapping['p50_latency']].mean(),
        proactive_data[column_mapping['p90_latency']].mean(),
        proactive_data[column_mapping['p99_latency']].mean()
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, reactive_values, width, label='Reactive Scaling')
    plt.bar(x + width/2, proactive_values, width, label='Proactive Scaling')
    plt.title('Key Metrics Comparison')
    plt.xticks(x, metrics)
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('metrics_comparison.png')
    
    print("\nCharts saved as: pods_comparison.png, latency_comparison.png, reactive_pods_requests.png, proactive_pods_requests.png, and metrics_comparison.png")

if __name__ == "__main__":
    reactive_file = "load_test_data_4_25_2025__2_01_47_am.csv"
    proactive_file = "load_test_data_4_26_2025__12_14_51_am.csv"
    
    try:
        results = analyze_scaler_data(reactive_file, proactive_file)
        
        print("\nSummary:")
        print(f"Proactive scaling uses {results['pod_increase_percentage']:.2f}% more pods on average")
        print(f"But achieves {abs(results['avg_reactive_latency'] - results['avg_proactive_latency']):.2f}ms lower average latency")
        print(f"P50 latency: {abs(results['p50_improvement_percentage']):.2f}% improvement")
        print(f"P90 latency: {abs(results['p90_improvement_percentage']):.2f}% improvement")
        print(f"P99 latency: {abs(results['p99_improvement_percentage']):.2f}% improvement")
        print(f"P99.9 latency: {abs(results['p99_9_improvement_percentage']):.2f}% improvement")
        print(f"P99.99 latency: {abs(results['p99_99_improvement_percentage']):.2f}% improvement")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV files exist in the current directory.")
