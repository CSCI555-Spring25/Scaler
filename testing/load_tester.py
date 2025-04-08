import schedule
import time
import subprocess
import datetime
import math
import random
import logging
import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Configuration
PEAK_HOURS = [8, 12, 18]  # Morning, midday, evening peaks
PEAK_WEIGHTS = [0.6, 0.8, 1.0]  # Relative weights of each peak
SIGMA_MINUTES = 60  # 1 hour standard deviation
MAX_REQUESTS = 100000  # Peak requests
ABSOLUTE_MAX_REQUESTS = 150000  # Absolute max requests
INTERVAL_MINUTES = 5  # Run test every 5 minutes
HISTORICAL_DATA_FILE = "historical_data.csv"
LOAD_TEST_LOG = "load_test.log"
POD_COUNTS_FILE = "pod_counts.json"
SEED = 42  # Set seed for repeatability
ENABLE_VISUALIZATION = True  # Generate graphs automatically

HOST_URL = "http://localhost:80"
HOST_ENDPOINT = "/" # requires at least / for ab testing cmd

# Set random seed for repeatability
random.seed(SEED)
np.random.seed(SEED)

# Configure logging
logging.basicConfig(filename=LOAD_TEST_LOG, level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def initialize_historical_data():
    if not os.path.exists(HISTORICAL_DATA_FILE):
        with open(HISTORICAL_DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = [
                "timestamp", "hour", "minute", "intended_requests",
                "concurrency_level", "time_taken", "complete_requests", "failed_requests",
                "total_transferred", "html_transferred", "requests_per_second",
                "time_per_request_mean", "time_per_request_all", "transfer_rate",
                "connect_min", "connect_mean", "connect_sd", "connect_median", "connect_max",
                "processing_min", "processing_mean", "processing_sd", "processing_median", "processing_max",
                "waiting_min", "waiting_mean", "waiting_sd", "waiting_median", "waiting_max",
                "total_min", "total_mean", "total_sd", "total_median", "total_max",
                "percentage_50", "percentage_66", "percentage_75", "percentage_80",
                "percentage_90", "percentage_95", "percentage_98", "percentage_99", "percentage_100"
            ]
            writer.writerow(headers)

def initialize_pod_counts():
    if not os.path.exists(POD_COUNTS_FILE):
        with open(POD_COUNTS_FILE, 'w') as f:
            json.dump({"data": []}, f)

def calculate_load(current_time):
    """
    Calculate requests based on multiple Gaussian distributions around PEAK_HOURS.
    """
    current_minutes = current_time.hour * 60 + current_time.minute
    
    # Calculate contribution from each peak hour
    total_load = 0
    for peak_hour, weight in zip(PEAK_HOURS, PEAK_WEIGHTS):
        peak_minutes = peak_hour * 60
        delta = current_minutes - peak_minutes
        
        # Wrap around for 24-hour cycles (e.g., late night is close to early morning)
        if delta > 12 * 60:
            delta = delta - 24 * 60
        elif delta < -12 * 60:
            delta = delta + 24 * 60
            
        # Gaussian calculation
        exponent = -(delta ** 2) / (2 * (SIGMA_MINUTES ** 2))
        gaussian = math.exp(exponent)
        total_load += gaussian * weight
    
    # Normalize the load to 0-1 scale
    total_load = min(total_load, 1.0)
    
    # Deterministic noise based on time
    noise_seed = int(current_time.timestamp()) % 1000
    r = np.random.RandomState(noise_seed)
    noise = r.uniform(-0.15, 0.15)  # +/- 15% noise
    
    # Calculate final request count
    base_requests = total_load * MAX_REQUESTS
    requests = int(base_requests * (1 + noise))
    requests = min(requests, ABSOLUTE_MAX_REQUESTS)  # Cap at absolute max
    
    return max(requests, 100)  # Ensure minimum requests

def parse_ab_output(output):
    # ... existing parse_ab_output function ...
    data = {
        'concurrency_level': 'N/A', 'time_taken': 'N/A',
        'complete_requests': '0', 'failed_requests': '0',
        'total_transferred': '0', 'html_transferred': '0',
        'requests_per_second': '0', 'time_per_request_mean': '0',
        'time_per_request_all': '0', 'transfer_rate': '0'
    }
    
    # Connection times and percentiles
    connection_categories = ['Connect', 'Processing', 'Waiting', 'Total']
    for cat in connection_categories:
        for metric in ['min', 'mean', 'sd', 'median', 'max']:
            data[f'{cat.lower()}_{metric}'] = '0'
    
    percentiles = ['50', '66', '75', '80', '90', '95', '98', '99', '100']
    for p in percentiles:
        data[f'percentage_{p}'] = '0'
    
    lines = output.split('\n')
    in_connection_times = False
    in_percentages = False

    for line in lines:
        line = line.strip()
        # Basic metrics
        if line.startswith("Concurrency Level:"):
            data['concurrency_level'] = line.split(':')[1].strip()
        elif line.startswith("Time taken for tests:"):
            data['time_taken'] = line.split()[4] if len(line.split()) >=5 else 'N/A'
        elif line.startswith("Complete requests:"):
            data['complete_requests'] = line.split(':')[1].strip()
        elif line.startswith("Failed requests:"):
            data['failed_requests'] = line.split(':')[1].strip()
        elif line.startswith("Total transferred:"):
            data['total_transferred'] = line.split(':')[1].split()[0].strip()
        elif line.startswith("HTML transferred:"):
            data['html_transferred'] = line.split(':')[1].split()[0].strip()
        elif line.startswith("Requests per second:"):
            data['requests_per_second'] = line.split()[3]
        elif line.startswith("Time per request:        "):
            if "(mean)" in line:
                data['time_per_request_mean'] = line.split()[3]
            elif "(mean, across all concurrent requests)" in line:
                data['time_per_request_all'] = line.split()[3]
        elif line.startswith("Transfer rate:"):
            data['transfer_rate'] = line.split()[2]
        # Connection Times
        elif line.startswith("Connection Times (ms)"):
            in_connection_times = True
        elif in_connection_times:
            if line.startswith("min  mean[+/-sd] median   max"):
                continue
            if not line:
                in_connection_times = False
                continue
            parts = line.split(':')
            if len(parts) < 2:
                continue
            category = parts[0].strip()
            values = parts[1].split()
            if len(values) >= 5:
                data[f'{category.lower()}_min'] = values[0]
                data[f'{category.lower()}_mean'] = values[1]
                data[f'{category.lower()}_sd'] = values[2]
                data[f'{category.lower()}_median'] = values[3]
                data[f'{category.lower()}_max'] = values[4]
        # Percentages
        elif line.startswith("Percentage of the requests served within a certain time (ms)"):
            in_percentages = True
        elif in_percentages:
            if not line:
                in_percentages = False
                continue
            if "%" in line:
                parts = line.split()
                for i in range(0, len(parts), 2):
                    if i+1 >= len(parts):
                        break
                    if '%' in parts[i]:
                        p = parts[i].replace('%', '')
                        data[f'percentage_{p}'] = parts[i+1]
    return data

def get_current_pod_count():
    try:
        # Get current pod count from kubectl
        cmd = "kubectl get pods -l app=simpleweb --no-headers | wc -l"
        result = subprocess.run(cmd, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=10)
        pod_count = int(result.stdout.strip())
        return pod_count
    except:
        # Return default if couldn't get count
        return 1

def update_pod_counts_json(pod_count):
    try:
        with open(POD_COUNTS_FILE, 'r') as f:
            data = json.load(f)
    except:
        data = {"data": []}
    
    # Add the current timestamp and pod count
    now = datetime.datetime.now()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"
    
    # Check if entry for this timestamp already exists
    updated = False
    for entry in data["data"]:
        if entry["timestamp"] == timestamp:
            entry["podCount"] = pod_count
            updated = True
            break
    
    # If no existing entry, add a new one
    if not updated:
        data["data"].append({"timestamp": timestamp, "podCount": pod_count})
    
    # Sort by timestamp
    data["data"].sort(key=lambda x: x["timestamp"])
    
    # Save updated data
    with open(POD_COUNTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def run_ab_test():
    now = datetime.datetime.now()
    requests = calculate_load(now)
    concurrency = max(requests // 100, 1)
    
    cmd = f"ab -n {requests} -c {concurrency} {HOST_URL}{HOST_ENDPOINT}"

    logging.info(f"Starting test: {cmd}")
    print(f"[{now}] Starting test: {cmd} (calculated load: {requests} requests)")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=300)
        parsed_data = parse_ab_output(result.stdout)
        
        # Record the test data
        with open(HISTORICAL_DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                now.timestamp(), now.hour, now.minute, requests,
                parsed_data['concurrency_level'], parsed_data['time_taken'],
                parsed_data['complete_requests'], parsed_data['failed_requests'],
                parsed_data['total_transferred'], parsed_data['html_transferred'],
                parsed_data['requests_per_second'], parsed_data['time_per_request_mean'],
                parsed_data['time_per_request_all'], parsed_data['transfer_rate'],
                parsed_data['connect_min'], parsed_data['connect_mean'],
                parsed_data['connect_sd'], parsed_data['connect_median'],
                parsed_data['connect_max'], parsed_data['processing_min'],
                parsed_data['processing_mean'], parsed_data['processing_sd'],
                parsed_data['processing_median'], parsed_data['processing_max'],
                parsed_data['waiting_min'], parsed_data['waiting_mean'],
                parsed_data['waiting_sd'], parsed_data['waiting_median'],
                parsed_data['waiting_max'], parsed_data['total_min'],
                parsed_data['total_mean'], parsed_data['total_sd'],
                parsed_data['total_median'], parsed_data['total_max'],
                parsed_data['percentage_50'], parsed_data['percentage_66'],
                parsed_data['percentage_75'], parsed_data['percentage_80'],
                parsed_data['percentage_90'], parsed_data['percentage_95'],
                parsed_data['percentage_98'], parsed_data['percentage_99'],
                parsed_data['percentage_100']
            ]
            writer.writerow(row)
        
        # Get and store the current pod count
        pod_count = get_current_pod_count()
        update_pod_counts_json(pod_count)
        
        # Generate visualizations if enabled
        if ENABLE_VISUALIZATION:
            try:
                subprocess.run(["python", "visualize.py"], check=False)
            except Exception as e:
                logging.error(f"Visualization error: {str(e)}")
        
        logging.info(f"Test completed: {cmd}")
        print(f"[{now}] Test completed: {requests} requests, {parsed_data['requests_per_second']} req/s, current pods: {pod_count}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Test failed: {e.stderr}")
        print(f"Test failed: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")

def visualize_predicted_load():
    """Generate a visualization of the predicted load pattern for the next 24 hours"""
    hours = []
    loads = []
    
    # Generate load for each 15-minute interval over 24 hours
    start_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    for hour in range(24):
        for minute in range(0, 60, 15):
            time_point = start_time.replace(hour=hour, minute=minute)
            hours.append(time_point)
            loads.append(calculate_load(time_point))
    
    # Create the visualization
    plt.figure(figsize=(12, 6))
    plt.plot(hours, loads, 'b-')
    plt.title('Predicted Load Pattern (Next 24 Hours)')
    plt.xlabel('Time of Day')
    plt.ylabel('Expected Requests')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.tight_layout()
    plt.savefig('predicted_load.png')
    plt.close()

if __name__ == "__main__":
    # Initialize data storage
    initialize_historical_data()
    initialize_pod_counts()
    
    # Generate and show the predicted load pattern
    visualize_predicted_load()
    print("Generated predicted load visualization (predicted_load.png)")
    
    # Run once immediately
    run_ab_test()
    
    # Schedule regular runs
    schedule.every(INTERVAL_MINUTES).minutes.do(run_ab_test)
    
    print(f"Load testing scheduled every {INTERVAL_MINUTES} minutes.")
    print(f"Using random seed {SEED} for repeatable pattern.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Load testing stopped by user.")