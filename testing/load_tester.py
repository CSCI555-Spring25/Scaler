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
MAX_REQUESTS = 10000  # Peak requests
ABSOLUTE_MAX_REQUESTS = 15000  # Absolute max requests
INTERVAL_MINUTES = 5  # Run test every 5 minutes
HISTORICAL_DATA_FILE = "historical_data.csv"
LOAD_TEST_LOG = "load_test.log"

HOST_URL = "http://localhost:80"
HOST_ENDPOINT = ""

# Configure logging
logging.basicConfig(filename=LOAD_TEST_LOG, level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def initialize_historical_data():
    if not os.path.exists(HISTORICAL_DATA_FILE):
        with open(HISTORICAL_DATA_FILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "hour", "minute", "requests", "errors"])

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
    """Extract RPS, latency, and errors from ab output."""
    lines = output.split('\n')
    rps = next((line for line in lines if "Requests per second" in line), "N/A")
    latency = next((line for line in lines if "Time per request" in line), "N/A")
    failed = int(next((line.split(':')[1].strip() for line in lines if "Failed requests" in line), 0))
    return rps, latency, failed

def run_ab_test():
    now = datetime.datetime.now()
    requests = calculate_load(now)
    concurrency = max(requests // 100, 1)
    
    cmd = f"ab -n {requests} -c {concurrency} {HOST_URL}{HOST_ENDPOINT}"
    logging.info(f"Starting test: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=300)
        rps, latency, errors = parse_ab_output(result.stdout)
        
        # Log metrics
        logging.info(f"Metrics:\n{rps}\n{latency}\nErrors: {errors}")
        
        # Save historical data
        with open(HISTORICAL_DATA_FILE, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                now.timestamp(),
                now.hour,
                now.minute,
                requests,
                errors
            ])
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Test failed: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

# Initialize files
initialize_historical_data()

# Schedule tests
schedule.every(INTERVAL_MINUTES).minutes.do(run_ab_test)

logging.info("Load testing started...")
while True:
    schedule.run_pending()
    time.sleep(1)