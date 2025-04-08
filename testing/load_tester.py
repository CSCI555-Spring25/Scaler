import schedule
import time
import subprocess
import datetime
import math
import random
import logging
import csv
import os

# Configuration
PEAK_HOUR = 18  # 6 PM
SIGMA_MINUTES = 60  # 1 hour standard deviation
MAX_REQUESTS = 1_000_000  # Peak requests
ABSOLUTE_MAX_REQUESTS = 1500  # Absolute max requests
INTERVAL_MINUTES = 5  # Run test every 5 minutes
HISTORICAL_DATA_FILE = "historical_data.csv"
LOAD_TEST_LOG = "load_test.log"
OVERIDE = True

HOST_URL = "http://localhost:80"
HOST_ENDPOINT = "/" # requires at least / for ab testing cmd

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

def calculate_load(current_time):
    """
    Calculate requests based on Gaussian distribution around PEAK_HOUR.
    - If current time is 5 PM, requests will be lower.
    - If current time is 6 PM, requests will be at maximum.
    Example:
    - 5 PM: 
        current_minutes = 17 * 60 + 0
        peak_minutes = 18 * 60
        delta = current_minutes - peak_minutes = -60
        exponent = -(delta ** 2) / (2 * (SIGMA_MINUTES ** 2)) = -(60 ** 2) / (2 * (60 ** 2)) = -0.5
        gaussian = math.exp(-0.5) = 0.6065306597126334
        base_requests = 0.6065306597126334 * 10000 = 6065
    - 6 PM:
        current_minutes = 18 * 60 + 0
        peak_minutes = 18 * 60
        delta = current_minutes - peak_minutes = 0
        exponent = -(delta ** 2) / (2 * (SIGMA_MINUTES ** 2)) = 0
        gaussian = math.exp(0) = 1
        base_requests = 1 * 10000 = 10000

    """
    current_minutes = current_time.hour * 60 + current_time.minute
    peak_minutes = PEAK_HOUR * 60
    delta = current_minutes - peak_minutes
    
    # Gaussian calculation
    exponent = -(delta ** 2) / (2 * (SIGMA_MINUTES ** 2))
    gaussian = math.exp(exponent)
    base_requests = gaussian * MAX_REQUESTS
    
    # Add ±20% noise
    noise = random.uniform(-0.2, 0.2)
    requests = int(base_requests * (1 + noise)) # +/- 20% noise
    requests = min(requests, ABSOLUTE_MAX_REQUESTS)  # Cap at absolute max
    return max(requests, 100)  # Ensure minimum requests

def parse_ab_output(output):
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

def run_ab_test():
    now = datetime.datetime.now()
    requests = calculate_load(now)
    concurrency = max(requests // 100, 1)
    
    cmd = f"ab -n {requests} -c {concurrency} {HOST_URL}{HOST_ENDPOINT}"
    if OVERIDE:
        requests = MAX_REQUESTS
        concurrency = 20_000
        cmd = f"ab -n {requests} -c {concurrency} {HOST_URL}{HOST_ENDPOINT}"


    logging.info(f"Starting test: {cmd}")
    print(f"Starting test: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=300)
        parsed_data = parse_ab_output(result.stdout)
        
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
        
        logging.info(f"Test completed: {cmd}")
        print(f"Test completed: {cmd}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Test failed: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")



# # Initialize files
# initialize_historical_data()

# # Schedule tests
# schedule.every(INTERVAL_MINUTES).minutes.do(run_ab_test)

# logging.info("Load testing started...")
# while True:
#     schedule.run_pending()
#     time.sleep(1)

if __name__ == "__main__":
    initialize_historical_data()
    run_ab_test()