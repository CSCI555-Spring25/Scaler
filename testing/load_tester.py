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
PEAK_HOURS = [2, 6, 7, 10, 14, 16, 18, 21]  # Multiple peaks at 8AM, 12PM, 6PM
PEAK_WEIGHTS = [random.uniform(0.9, 1.1) for _ in range(len(PEAK_HOURS))]
SIGMA_MINUTES = 40  # Peak width
MAX_RATE = 110       # Maximum requests/sec
MIN_RATE = 10        # Minimum requests/sec
THREADS = 1
CONNECTIONS = 4
DURATION_SECONDS = 60  # Test duration in seconds
# INTERVAL_MINUTES = 1    # 
HISTORICAL_DATA_FILE = "load_test_data.csv"
LOAD_TEST_LOG = "load_test.log"
OUTPUT_DIR = "./output"
HISTORICAL_DATA_FILE = os.path.join(OUTPUT_DIR, HISTORICAL_DATA_FILE)
LOAD_TEST_LOG = os.path.join(OUTPUT_DIR, LOAD_TEST_LOG)
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
OVERIDE = False

# traffic configuration


HOST_URL = "http://128.105.146.155/"

# Configure logging
logging.basicConfig(filename=LOAD_TEST_LOG, level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def initialize_historical_data():
    HEADERS = ["timestamp", "hour", "minute", "pods", "target_rate",
                "threads", "connections", "duration_sec",
                "total_requests", "data_transferred_mb",
                "requests_per_sec", "transfer_per_sec_mb",
                "latency_avg_ms", "latency_stdev_ms", "latency_max_ms", 
                "latency_stdev_pct", "req_sec_avg", "req_sec_stdev",
                "req_sec_max", "req_sec_stdev_pct",
                "p50_ms", "p75_ms", "p90_ms", "p99_ms", 
                "p99.9_ms", "p99.99_ms", "p99.999_ms", "p100_ms"]
    if not os.path.exists(HISTORICAL_DATA_FILE):
        with open(HISTORICAL_DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = HEADERS
            writer.writerow(headers)
    else:
        logging.info(f"Historical data file '{HISTORICAL_DATA_FILE}' already exists.")
        # verify headers are correct
        with open(HISTORICAL_DATA_FILE, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            if headers != HEADERS:
                logging.error(f"Historical data file '{HISTORICAL_DATA_FILE}' headers do not match expected format.")
                raise ValueError("Historical data file headers do not match expected format.")

def get_hpa_info():
    """
    return tuple of (pod_count, desiredReplicas, currentReplicas, nodes_count)
    """
    # pod_count=$(kubectl get pods -l app=simpleweb -o json | jq '.items | length')
    # hpa_status=$(kubectl get hpa simpleweb-hpa -o json | jq -c '[.status.desiredReplicas, .status.currentReplicas]')
    # nodes_count=$(kubectl get nodes --no-headers | wc -l)
    # the above works in bash, now do python
    try:
        pod_count = subprocess.check_output("kubectl get pods -l app=simpleweb -o json | jq '.items | length'", shell=True)
        pod_count = int(pod_count.strip())
        hpa_status = subprocess.check_output("kubectl get hpa simpleweb-hpa -o json | jq -c '[.status.desiredReplicas, .status.currentReplicas]'", shell=True)
        hpa_status = hpa_status.decode('utf-8').strip().split(',')
        desired_replicas = int(hpa_status[0].strip('[]'))
        current_replicas = int(hpa_status[1].strip('[]'))
        nodes_count = subprocess.check_output("kubectl get nodes --no-headers | wc -l", shell=True)
        nodes_count = int(nodes_count.strip())
        return pod_count, desired_replicas, current_replicas, nodes_count
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting HPA info: {e}")
        return 0, 0, 0, 0

def calculate_traffic_rate(current_time):
    """Multi-peak traffic calculation"""
    current_minutes = current_time.hour * 60 + current_time.minute
    combined_rate = 0.0
    
    for peak_hour, weight in zip(PEAK_HOURS, PEAK_WEIGHTS):
        peak_minutes = peak_hour * 60
        delta = current_minutes - peak_minutes
        exponent = -(delta ** 2) / (2 * (SIGMA_MINUTES ** 2))
        peak_contribution = weight * math.exp(exponent)
        # Normalize to MAX_RATE * weight at the peak
        combined_rate += peak_contribution * MAX_RATE

    # Add noise
    noise = random.uniform(-0.1, 0.1)
    rate = int(combined_rate * (1 + noise))

    return max(rate, MIN_RATE)

def parse_wrk2_output(output):
    data = {
        'total_requests': 0, 'data_transferred_mb': 0.0,
        'requests_per_sec': 0.0, 'transfer_per_sec_mb': 0.0,
        'latency_avg_ms': 0.0, 'latency_stdev_ms': 0.0,
        'latency_max_ms': 0.0, 'latency_stdev_pct': 0.0,
        'req_sec_avg': 0.0, 'req_sec_stdev': 0.0,
        'req_sec_max': 0.0, 'req_sec_stdev_pct': 0.0,
        'p50_ms': 0.0, 'p75_ms': 0.0, 'p90_ms': 0.0,
        'p99_ms': 0.0, 'p99.9_ms': 0.0, 'p99.99_ms': 0.0,
        'p99.999_ms': 0.0, 'p100_ms': 0.0
    }
    
    lines = output.split('\n')
    in_percentiles = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('Requests/sec:'):
            data['requests_per_sec'] = float(line.split()[1])
        elif line.startswith('Transfer/sec:'):
            transfer = line.split()[1].replace('MB', '').replace('KB', 'e-3')
            data['transfer_per_sec_mb'] = float(transfer)
        elif line.startswith('Latency   '):
            parts = line.split()
            data['latency_avg_ms'] = parse_time(parts[1])
            data['latency_stdev_ms'] = parse_time(parts[2])
            data['latency_max_ms'] = parse_time(parts[3])
            data['latency_stdev_pct'] = float(parts[4].replace('%', ''))
        elif line.startswith('Req/Sec  '):
            parts = line.split()
            data['req_sec_avg'] = float(parts[1])
            data['req_sec_stdev'] = float(parts[2])
            data['req_sec_max'] = float(parts[3])
            data['req_sec_stdev_pct'] = float(parts[4].replace('%', ''))
        elif 'requests in ' in line and 'read' in line:
            # Handle lines like: "4800 requests in 1.00m, 30.95MB read"
            clean_line = line.replace(',', ' ')  # Remove comma from duration
            parts = clean_line.split()
            data['total_requests'] = int(parts[0])
            data['data_transferred_mb'] = float(parts[4].replace('MB', ''))
        elif line.startswith('Latency Distribution'):
            in_percentiles = True
        elif in_percentiles and '%' in line:
            percentile, value = line.split('%')
            # print(f"Parsing percentile: {percentile.strip()} with value: {value.strip()}")
            # strip and remove trailing 0s
            percentile = float(percentile.strip())
            # if percentile == int(percentile):
            #     percentile = str(int(percentile))
            # else:
            percentile = str(percentile).rstrip('0').rstrip('.')
            value = parse_time(value.strip().split()[0])
            # print(f"Parsing percentile: {percentile} with value: {value}")
            data_key = f'p{percentile}_ms' if percentile != '100' else 'p100_ms'
            # print(f"Data key: {data_key}")
            if data_key in data:
                data[data_key] = value
                
    return data

def parse_time(value):
    units = {'ms': 1, 's': 1000, 'us': 0.001, 'm': 60000}  # Added minute support
    for unit, multiplier in units.items():
        if value.endswith(unit):
            return float(value[:-len(unit)]) * multiplier
    return float(value)

def run_load_test():
    now = datetime.datetime.now()
    rate = calculate_traffic_rate(now)
    test_duration = DURATION_SECONDS
    
    if OVERIDE:
        rate = 80  # Fixed rate for dry run
        test_duration = 60  # Local variable instead of modifying global

    cmd = f"wrk2 -t{THREADS} -c{CONNECTIONS} -d{test_duration}s -R{rate} --latency {HOST_URL}"
    
    # logging.info(f"Starting test: {cmd}")
    logging.info(f"{now.hour}:{now.minute} rate: {rate} pods: {get_hpa_info()[0]}")
    print(f"{now.hour}:{now.minute}:{now.second} rate: {rate} pods: {get_hpa_info()[0]}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=test_duration+8)
        
        # Log raw output for debugging
        logging.info(f"WRK2 Output:\n{result.stdout}")
        if result.stderr:
            logging.error(f"WRK2 Errors:\n{result.stderr}")

        parsed = parse_wrk2_output(result.stdout)
        # pods_count = get_pods_count() 
        pods_count, desired_replicas, current_replicas, nodes_count = get_hpa_info()
        
        with open(HISTORICAL_DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                now.timestamp(), now.hour, now.minute, pods_count, rate,
                THREADS, CONNECTIONS, test_duration,
                parsed['total_requests'], parsed['data_transferred_mb'],
                parsed['requests_per_sec'], parsed['transfer_per_sec_mb'],
                parsed['latency_avg_ms'], parsed['latency_stdev_ms'],
                parsed['latency_max_ms'], parsed['latency_stdev_pct'],
                parsed['req_sec_avg'], parsed['req_sec_stdev'],
                parsed['req_sec_max'], parsed['req_sec_stdev_pct'],
                parsed['p50_ms'], parsed['p75_ms'], parsed['p90_ms'],
                parsed['p99_ms'], parsed['p99.9_ms'], parsed['p99.99_ms'],
                parsed['p99.999_ms'], parsed['p100_ms']
            ])
        
        logging.info(f"Test completed: {cmd}")
        print(f"Test completed: {cmd}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Test failed: {e.stderr}")
    except Exception as e:
        logging.error(f"Error: {str(e)}")

# Initialize data file
initialize_historical_data()

# Uncomment for scheduled operation
# schedule.every(INTERVAL_MINUTES).minutes.do(run_load_test)
# schedule.every().day.at("00:00").do(run_load_test)


# while True:
#     schedule.run_pending()
#     time.sleep(1)

if __name__ == "__main__":
    while True:
        run_load_test()