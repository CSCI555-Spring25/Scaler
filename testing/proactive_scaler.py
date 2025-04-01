import csv
import datetime
import os
import subprocess
import time

HISTORICAL_DATA_FILE = "historical_data.csv"
LOOKBACK_DAYS = 3  # Analyze past 3 days of data

def load_historical_data():
    data = {}
    if not os.path.exists(HISTORICAL_DATA_FILE):
        return data
    
    with open(HISTORICAL_DATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hour = int(row['hour'])
            minute = int(row['minute'])
            requests = int(row['requests'])
            
            # Group by 5-minute windows
            time_key = f"{hour}:{minute//5*5:02d}"
            if time_key not in data:
                data[time_key] = []
            data[time_key].append(requests)
    
    # Calculate averages
    averages = {}
    for key, values in data.items():
        averages[key] = sum(values) / len(values)
    return averages

def calculate_desired_pods(current_pods, historical_avg):
    """Apply scaling formula: min((current_pods/historical_pods) * future_pods, max_pods)"""
    max_pods = 10  # Set your HPA max_pods
    if historical_avg == 0:
        return 1  # Minimum pods
    
    # Example: If historical shows 2000 requests in 10 mins, scale proportionally
    scaling_factor = (current_pods / historical_avg) * historical_avg
    return min(int(scaling_factor), max_pods)

def scale_deployment():
    now = datetime.datetime.now()
    current_time_key = f"{now.hour}:{(now.minute//5)*5:02d}"
    
    historical_data = load_historical_data()
    historical_value = historical_data.get(current_time_key, 0)
    
    # Get current pod count
    current_pods = subprocess.check_output(
        "kubectl get deployment simpleweb-deployment -o jsonpath='{.status.readyReplicas}'",
        shell=True
    ).decode().strip()
    current_pods = int(current_pods) if current_pods else 1
    
    desired_pods = calculate_desired_pods(current_pods, historical_value)
    
    # Scale deployment
    subprocess.run(
        f"kubectl scale deployment simpleweb-deployment --replicas={desired_pods}",
        shell=True
    )
    print(f"[{now}] Scaled to {desired_pods} pods (historical avg: {historical_value})")

if __name__ == "__main__":
    while True:
        scale_deployment()
        time.sleep(300)  # Run every 5 minutes