#!/usr/bin/python3
import requests
import time
import statistics
import matplotlib.pyplot as plt

# Configuration
SERVER_URL = "http://130.127.132.204"
REQUESTS_COUNT = 250
REQUEST_INTERVAL = 0.1  # interval between requests in seconds
HISTOGRAM_BINS = 10  # number of bins for latency histogram


def measure_latency(url, requests_count, interval):
    latencies = []
    tasks = []

    for i in range(requests_count):
        try:
            start_time = time.perf_counter()
            response = requests.get(url)
            latency = time.perf_counter() - start_time

            latencies.append(latency)
            task_info = "No task executed"

            if response.ok:
                content = response.text
                if "Compute-intensive task executed:" in content:
                    for line in content.split('\n'):
                        if line.startswith("Compute-intensive task executed:"):
                            task_info = line.replace("Compute-intensive task executed: ", "").strip()
                            break

            tasks.append((task_info, latency))
            print(f"Request {i+1}: status {response.status_code}, latency {latency:.4f} sec, task: {task_info}")

        except requests.exceptions.RequestException as e:
            print(f"Request {i+1} failed: {e}")
            continue

        time.sleep(interval)

    return latencies, tasks


def analyze_latencies(latencies):
    return {
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "mean_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "stdev_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


def task_summary(tasks):
    summary = {}
    task_latencies = {}
    for task, latency in tasks:
        summary[task] = summary.get(task, 0) + 1
        task_latencies.setdefault(task, []).append(latency)
    return summary, task_latencies


def plot_task_latencies(task_latencies):
    """
    Plot and save per-request latency bar charts for each task.
    """
    for task, latencies in task_latencies.items():
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, len(latencies) + 1), latencies)
        plt.title(f'Latency per Request for {task}')
        plt.xlabel('Request Number')
        plt.ylabel('Latency (s)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename = f'latency_{task.replace(" ", "_").lower()}.png'
        plt.savefig(f"./plots/{filename}")
        print(f'Saved per-request plot for {task} as {filename}')
        plt.close()


def plot_latency_histograms(task_latencies, bins):
    """
    Plot and save histogram of latencies for each task with uniform bins.
    """
    for task, latencies in task_latencies.items():
        plt.figure(figsize=(8, 4))
        plt.hist(latencies, bins=bins, edgecolor='black')
        plt.title(f'Latency Histogram for {task} ({bins} bins)')
        plt.xlabel('Latency (s)')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        hist_filename = f'histogram_{task.replace(" ", "_").lower()}.png'
        plt.savefig(f"./plots/{hist_filename}")
        print(f'Saved histogram for {task} as {hist_filename}')
        plt.close()


def plot_combined_charts(latencies, tasks, bins):
    """
    Plot and save combined charts across all tasks.
    """
    # Combined per-request latency bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(latencies) + 1), latencies)
    plt.title('Combined Latency per Request (All Tasks)')
    plt.xlabel('Request Number')
    plt.ylabel('Latency (s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    combined_bar = 'combined_latency_per_request.png'
    plt.savefig(f"./plots/{combined_bar}")
    print(f'Saved combined per-request plot as {combined_bar}')
    plt.close()

    # Combined histogram
    plt.figure(figsize=(10, 5))
    plt.hist(latencies, bins=bins, edgecolor='black')
    plt.title(f'Combined Latency Histogram ({bins} bins)')
    plt.xlabel('Latency (s)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    combined_hist = 'combined_latency_histogram.png'
    plt.savefig(f"./plots/{combined_hist}")
    print(f'Saved combined histogram as {combined_hist}')
    plt.close()


def main():
    print("Starting latency measurement...")
    latencies, tasks = measure_latency(SERVER_URL, REQUESTS_COUNT, REQUEST_INTERVAL)

    if not latencies:
        print("No successful requests were recorded.")
        return

    # Overall stats
    stats = analyze_latencies(latencies)
    print("\nOverall Latency Statistics:")
    print(f"  Min: {stats['min_latency']:.4f} s")
    print(f"  Max: {stats['max_latency']:.4f} s")
    print(f"  Mean: {stats['mean_latency']:.4f} s")
    print(f"  Median: {stats['median_latency']:.4f} s")
    print(f"  Std Dev: {stats['stdev_latency']:.4f} s")

    # Task summary and task-specific stats
    task_stats, task_latencies = task_summary(tasks)
    print("\nTask Execution Summary:")
    for task, count in task_stats.items():
        task_specific_latencies = task_latencies[task]
        task_analysis = analyze_latencies(task_specific_latencies)
        print(f"\n{task} executed {count} times")
        print(f"  - Min Latency: {task_analysis['min_latency']:.4f} sec")
        print(f"  - Max Latency: {task_analysis['max_latency']:.4f} sec")
        print(f"  - Mean Latency: {task_analysis['mean_latency']:.4f} sec")
        print(f"  - Median Latency: {task_analysis['median_latency']:.4f} sec")
        print(f"  - Std Dev Latency: {task_analysis['stdev_latency']:.4f} sec")

    # Plot and save charts
    print("\nSaving per-request latency bar charts...")
    plot_task_latencies(task_latencies)

    print("\nSaving latency histograms...")
    plot_latency_histograms(task_latencies, HISTOGRAM_BINS)

    print("\nSaving combined charts for all tasks...")
    plot_combined_charts(latencies, tasks, HISTOGRAM_BINS)

if __name__ == "__main__":
    main()
