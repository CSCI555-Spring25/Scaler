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
import traceback

POLL_INTERVAL = 1  # seconds

PST = pytz.timezone('US/Pacific')
now_pst = datetime.datetime.now(PST)
timestamp_str = now_pst.strftime("%-m_%-d_%Y__%-I_%M_%S_%p").lower()

# Configuration
fall_sigma_min = 25
plateau_min = 20
# Configuration
PEAK_PARAMS = [
  # (hour, weight, rise_sigma_min, fall_sigma_min, plateau_min)
    (1*60,  1.0,       1,             fall_sigma_min,  plateau_min ),
    (4*60,  0.98,      2,             fall_sigma_min,  plateau_min ),   # e.g. 7AM peak, fall_sigma_min‑min rise, 20‑min flat, fall_sigma_min‑min fall
    (7*60,  1.0,       5,             fall_sigma_min,  plateau_min ),   
    (10*60,  1.0,      8,             fall_sigma_min,  plateau_min ),  
    (13*60,  1.0,      14,             fall_sigma_min,  plateau_min ),   
    (16*60, 1.0,      18,             fall_sigma_min,  plateau_min ),
    (19*60, 1.0,      21,             fall_sigma_min,  plateau_min ),
    (22*60, 1.0,      24,             fall_sigma_min,  plateau_min ),
]

MAX_RATE = 620       # Maximum requests/sec
MIN_RATE = 30        # Minimum requests/sec


# overprovision
THREADS = 1
CONNECTIONS = 8
DURATION_SECONDS = 60  # Test duration in seconds
# INTERVAL_MINUTES = 1    # 
# traffic configuration
HISTORICAL_DATA_FILE = f"load_test_data_{timestamp_str}.csv"
LOAD_TEST_LOG = f"load_test_{timestamp_str}.log"
POD_INFO_CSV = f"pod_scaling_samples_{timestamp_str}.csv"
OUTPUT_DIR = "./output"
LOAD_TEST_OUTPUT_DIR = OUTPUT_DIR+ "/loads"
LOG_OUTPUT_DIR = OUTPUT_DIR + "/log"
POD_OUTPUT_DIR = OUTPUT_DIR + "/pod_info"

HISTORICAL_DATA_FILE = os.path.join(LOAD_TEST_OUTPUT_DIR, HISTORICAL_DATA_FILE)
LOAD_TEST_LOG = os.path.join(LOG_OUTPUT_DIR, LOAD_TEST_LOG)
POD_INFO_CSV = os.path.join(POD_OUTPUT_DIR, POD_INFO_CSV)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOAD_TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)
os.makedirs(POD_OUTPUT_DIR, exist_ok=True)

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

def compute_scale_up_latencies(samples):
    scale_events = []
    last_desired = None

    for entry in samples:
        t = entry['timestamp']
        ready = entry['ready_pods']
        desired = entry['desired_replicas']

        if last_desired is not None and desired > last_desired:
            # HPA triggered scale-up
            scale_events.append({
                'scale_time': t,
                'desired': desired,
                'ready_at': None
            })
        last_desired = desired

        # Fill in "ready_at" once enough pods are ready
        for event in scale_events:
            if event['ready_at'] is None and ready >= event['desired']:
                event['ready_at'] = t

    # Calculate latencies
    for event in scale_events:
        if event['ready_at']:
            latency = event['ready_at'] - event['scale_time']
            print(f"Scale to {event['desired']} pods: took {latency:.2f} sec to become ready")
        else:
            print(f"Scale to {event['desired']} pods: never fully ready during test")

def get_ready_pods_count() -> int:
    cmd = "kubectl get pods -l app=simpleweb -o json | jq '[.items[] | select(.status.phase==\"Running\") | select(all(.status.containerStatuses[]; .ready==true))] | length'"
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    return int(output)

def get_hpa_status() -> tuple[int, int]:
    cmd = "kubectl get hpa simpleweb-hpa -o json | jq -c '[.status.desiredReplicas, .status.currentReplicas]'"
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split(',')
    desired = int(output[0].strip('[]'))
    current = int(output[1].strip('[]'))
    return current, desired

def get_hpa_info() -> tuple[int, int, int]:
    """
    return tuple of (pod_count, desiredReplicas, currentReplicas, nodes_count)
    """
    try:
        pod_count = get_ready_pods_count()
        current_replicas, desired_replicas = get_hpa_status()
        return pod_count, desired_replicas, current_replicas
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting HPA info: {e}")
        return 0, 0, 0, 0

def sample_pod_counts(duration, interval, samples, test_start_time):
    start_time = time.time()
    while time.time() - start_time < duration:
        timestamp = time.time()
        pod_count, desired_replicas, current_replicas = get_hpa_info()
        entry = {
            'timestamp': timestamp,
            'ready_pods': pod_count,
            'desired_replicas': desired_replicas,
            'current_replicas': current_replicas
        }
        samples.append(entry)
        save_pod_sample_to_csv(test_start_time, entry)
        time.sleep(interval)

def save_pod_sample_to_csv(test_start_time, entry):
    """
    Append a single pod sample entry to the CSV during test execution.
    """
    filename = POD_INFO_CSV
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            test_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
            entry['ready_pods'],
            entry['desired_replicas'],
            entry['current_replicas']
        ])

def init_pod_samples_csv():
    """
    Initialize the CSV file by writing headers if it doesn't already exist.
    """
    filename = POD_INFO_CSV
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "test_start_time",
                "sample_time",
                "ready_pods",
                "desired_replicas",
                "current_replicas"
            ])

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
            data['transfer_per_sec_mb'] = parse_with_units(line.split()[1])
        elif line.startswith('Latency   '):
            parts = line.split()
            data['latency_avg_ms'] = parse_time(parts[1])
            data['latency_stdev_ms'] = parse_time(parts[2])
            data['latency_max_ms'] = parse_time(parts[3])
            data['latency_stdev_pct'] = float(parts[4].replace('%', ''))
        elif line.startswith('Req/Sec  '):
            parts = line.split()
            data['req_sec_avg'] = parse_with_units(parts[1])
            data['req_sec_stdev'] = parse_with_units(parts[2])
            data['req_sec_max'] = parse_with_units(parts[3])
            data['req_sec_stdev_pct'] = float(parts[4].replace('%', ''))
        elif 'requests in ' in line and 'read' in line:
            # Handle lines like: "4800 requests in 1.00m, 30.95MB read"
            clean_line = line.replace(',', ' ')  # Remove comma from duration
            parts = clean_line.split()
            data['total_requests'] = int(parts[0])
            data['data_transferred_mb'] = parse_with_units(parts[4])
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

def parse_with_units(s):
    try:
        s = s.strip().upper().replace(',', '')
        if s.endswith('MB'):
            return round(float(s[:-2]), 2)
        elif s.endswith('KB'):
            return round(float(s[:-2]) * 1e-3, 2)
        elif s.endswith('K'):
            return round(float(s[:-1]) * 1e3, 2)
        elif s.endswith('M'):
            return round(float(s[:-1]) * 1e6, 2)
        elif s.endswith('B'):
            return round(float(s[:-1]), 2)
        else:
            return round(float(s), 2)
    except Exception as e:
        print_and_log("An unexpected error occurred during load test.")
        print_and_log(f"Exception: {repr(e)}")
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print_and_log(f"Full traceback:\n{tb}")
        return 0

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

def calculate_weighted_pod_average(samples):
    if len(samples) < 2:
        return samples[0]['ready_pods'] if samples else 0

    weighted_sum = 0
    total_time = 0
    for i in range(1, len(samples)):
        duration = samples[i]['timestamp'] - samples[i-1]['timestamp']
        weighted_sum += samples[i-1]['ready_pods'] * duration
        total_time += duration

    weighted_avg = round(weighted_sum / total_time, 2)
    return weighted_avg if total_time > 0 else 0

