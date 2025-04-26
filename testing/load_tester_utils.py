import subprocess
import datetime
import logging
import csv
import os
import pytz
import time
import math
import random
import threading

POLL_INTERVAL = 1  # seconds

PST = pytz.timezone('US/Pacific')
now_pst = datetime.datetime.now(PST)
timestamp_str = now_pst.strftime("%-m_%-d_%Y__%-I_%M_%S_%p").lower()

# Configuration
fall_sigma_min = 12
plateau_min = 12

PEAK_PARAMS = [
    (i * 60, 1.0, i, fall_sigma_min - (i // 2.5), plateau_min - (i // 2.5))
    for i in range(0, 25)
]

# generates:
# PEAK_PARAMS = [
#     # (hour, weight, rise_sigma_min, fall_sigma_min, plateau_min)
#     (1*60, 1.0,  1,  fall_sigma_min,  plateau_min ),   # e.g. 7AM peak, fall_sigma_min‑min rise, 20‑min flat, fall_sigma_min‑min fall
#     (2*60, 1.0,  2, fall_sigma_min,  plateau_min ),   # noon
#     (3*60, 1.0,  3,  fall_sigma_min,  plateau_min ),   # 6PM
#     ...
# ]
MAX_RATE = 105       # Maximum requests/sec
MIN_RATE = 8        # Minimum requests/sec

# overprovision
THREADS = 1
CONNECTIONS = 8
DURATION_SECONDS = 60  # Test duration in seconds
# INTERVAL_MINUTES = 1    # 
# traffic configuration
HISTORICAL_DATA_FILE = f"load_test_data_{timestamp_str}.csv"
LOAD_TEST_LOG = f"load_test_{timestamp_str}.log"
OUTPUT_DIR = "./output"
HISTORICAL_DATA_FILE = os.path.join(OUTPUT_DIR, HISTORICAL_DATA_FILE)
LOAD_TEST_LOG = os.path.join(OUTPUT_DIR, LOAD_TEST_LOG)
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def custom_time(*args):
    return datetime.now(PST).timetuple()

logging.Formatter.converter = custom_time

# Configure logging
logging.basicConfig(filename=LOAD_TEST_LOG, 
                    level=logging.INFO,
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

_noise_prev = 0.0
def correlated_noise(alpha=0.9, scale=0.015):
    global _noise_prev
    e = random.gauss(0, scale)
    _noise_prev = alpha*_noise_prev + e
    return _noise_prev

def make_noise():
    min_val = -0.07
    max_val = 0.07
    return random.uniform(min_val, max_val)

def baseline_multiplier(dt):
    # weekends 40% lower
    # return 0.6 if dt.weekday() >= 5 else 1.0
    return 1.0

def calculate_traffic_rate(dt: datetime):
    t_min = dt.hour*60 + dt.minute
    total_weight = 0.0

    for peak_min, weight, rise_s, fall_s, plateau in PEAK_PARAMS:
        total_weight += peak_contribution(
            t_min, peak_min, weight,
            rise_s, fall_s, plateau
        )

    # Scale to rate, apply baseline and noise
    # noise = correlated_noise()
    noise = make_noise()
    rate = total_weight * MAX_RATE * baseline_multiplier(dt) * (1 + noise)
    rate = max(int(rate), MIN_RATE)
    if rate < MIN_RATE * 2:
        rate = rate * (1 + noise + 0.15) ** 3
    return max(int(rate), MIN_RATE)

def peak_contribution(t_min, peak_min, weight,
                      rise_sigma, fall_sigma, plateau):
    dt = t_min - peak_min
    # plateau region
    if abs(dt) <= plateau/2:
        return weight
    # rising edge
    if -plateau/2 - 3*rise_sigma < dt < -plateau/2:
        x = dt + plateau/2
        return weight * math.exp(-x*x/(2*rise_sigma**2))
    # falling edge
    if plateau/2 < dt < plateau/2 + 3*fall_sigma:
        x = dt - plateau/2
        return weight * math.exp(-x*x/(2*fall_sigma**2))
    return 0.0


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
            data['data_transferred_mb'] = float(parts[4].replace('MB', '').replace('KB', 'e-3'))
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
    units = {'us': 0.001, 'ms': 1, 's': 1000, 'm': 60000}  # Added minute support
    for unit, multiplier in units.items():
        if value.endswith(unit):
            return float(value[:-len(unit)]) * multiplier
    return float(value)

def print_and_log(p):
    logging.info(p)
    print(p)

import csv
from datetime import datetime

def write_historical_data(
    HISTORICAL_DATA_FILE: str,
    now: datetime,
    pods_count: int,
    rate: float,
    THREADS: int,
    CONNECTIONS: int,
    test_duration: int,
    parsed: dict
):
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

def sample_pod_counts(duration, interval, samples):
    start_time = time.time()
    while time.time() - start_time < duration:
        timestamp = time.time()
        pod_count = get_hpa_info()[0]
        samples.append((timestamp, pod_count))
        time.sleep(interval)

def calculate_weighted_pod_average(samples):
    if len(samples) < 2:
        return samples[0][1] if samples else 0

    weighted_sum = 0
    total_time = 0
    for i in range(1, len(samples)):
        duration = samples[i][0] - samples[i-1][0]
        weighted_sum += samples[i-1][1] * duration
        total_time += duration
    
    weighted_avg = round(weighted_sum / total_time, 2)
    return weighted_avg if total_time > 0 else 0
