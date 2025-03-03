#!/usr/bin/env python3
import kopf
import kubernetes
import os
import json
import datetime
import time
import threading
import logging

# Load Kubernetes configuration
kubernetes.config.load_incluster_config()

# Initialize clients
api_client = kubernetes.client.ApiClient()
apps_api = kubernetes.client.AppsV1Api(api_client)
autoscaling_api = kubernetes.client.AutoscalingV2Api(api_client)
custom_api = kubernetes.client.CustomObjectsApi(api_client)

# Constants
DATA_DIR = "/data"
GROUP = "scaler.cs.usc.edu"
VERSION = "v1"
PLURAL = "predictiveautoscalers"

logger = logging.getLogger(__name__)
thread_logger = logging.getLogger(f"{__name__}.thread")

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def get_history_file_path(namespace=None, name=None):
    if not namespace and not name: #to be used for testing
        return os.path.join(DATA_DIR, "realistic-traffic.json") 
    else:
        return os.path.join(DATA_DIR, f"{namespace}_{name}_history.json")

def load_historical_data(namespace=None, name=None):
    file_path = get_history_file_path(namespace, name)
    if not os.path.exists(file_path):
        return {"data": []}
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"data": []}

def save_historical_data(data, namespace=None, name=None):
    ensure_data_dir()
    file_path = get_history_file_path(namespace, name)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_current_pod_count(namespace, deployment_name):
    try:
        deployment = apps_api.read_namespaced_deployment(deployment_name, namespace)
        return deployment.status.replicas or 0
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 404:
            logger.info(f"Deployment {deployment_name} not found in namespace {namespace}")
            return 0
        raise

def get_historical_pod_count_at_time(data, time_offset_minutes=0):
    now = datetime.datetime.now(datetime.timezone.utc)
    target_time = now - datetime.timedelta(minutes=time_offset_minutes)
    
    # Find the closest historical data point - floor to 5-minute interval
    target_hour = target_time.hour
    target_minute = (target_time.minute // 5) * 5  # Floor to nearest 5-minute interval
    
    # Create target timestamp in HH:MM format
    target_timestamp = f"{target_hour:02d}:{target_minute:02d}"
    logger.info(f"Looking for historical data at {target_timestamp}")
    
    # First try to find exact match
    for entry in data["data"]:
        if entry["timestamp"] == target_timestamp:
            logger.info(f"Found exact match for {target_timestamp} with pod count {entry['podCount']}")
            return entry["podCount"]
    
    # If exact match not found, find closest timestamp that's earlier
    # Convert target to minutes since midnight for easy comparison
    target_minutes = target_hour * 60 + target_minute
    
    closest_entry = None
    closest_diff = float('inf')
    
    for entry in data["data"]:
        # Parse HH:MM format
        h, m = map(int, entry["timestamp"].split(':'))
        entry_minutes = h * 60 + m
        
        # Only consider earlier timestamps
        if entry_minutes <= target_minutes:
            diff = target_minutes - entry_minutes
            if diff < closest_diff:
                closest_diff = diff
                closest_entry = entry
    
    if closest_entry:
        logger.info(f"Found closest earlier timestamp {closest_entry['timestamp']} with pod count {closest_entry['podCount']}")
        return closest_entry["podCount"]
    
    # Default if not found
    logger.info(f"No historical data found for {target_timestamp}, using default pod count 1")
    return 1

def update_historical_data(data, current_pods, historical_weight=0.7, current_weight=0.3):
    now = datetime.datetime.now(datetime.timezone.utc)
    # Format timestamp as HH:MM
    timestamp = f"{now.hour:02d}:{(now.minute // 5) * 5:02d}"
    
    # Find if we have an entry for this timestamp
    for entry in data["data"]:
        if entry["timestamp"] == timestamp:
            historical_count = entry["podCount"]
            # Update using the given formula
            entry["podCount"] = int(historical_weight * historical_count + current_weight * current_pods)
            logger.info(f"Updated historical data for timestamp {timestamp} to {entry['podCount']}")
            break
    else:
        # No entry found, create new one
        data["data"].append({"timestamp": timestamp, "podCount": current_pods})
        logger.info(f"Created new historical data entry for timestamp {timestamp} with pod count {current_pods}")
    
    # Sort by timestamp - for HH:MM format 
    # Convert to minutes for proper sorting
    def timestamp_to_minutes(ts):
        h, m = map(int, ts.split(':'))
        return h * 60 + m
    
    data["data"] = sorted(data["data"], key=lambda x: timestamp_to_minutes(x["timestamp"]))
    
    return data

def prune_old_data(data, retention_days=7):
    # Since we're using simplified timestamps without dates, 
    # we can't easily prune by date anymore
    # This function could be modified to keep a maximum number of entries
    # or all entries could be kept since they're just HH:MM timestamps
    
    # For now, we'll keep all entries since the set is limited (288 entries for a day)
    logger.info(f"Historical data has {len(data['data'])} entries")
    return data

def calculate_required_pods(current_pods, historical_data, max_replicas, prediction_window_minutes=10):
    # Get historical pod count at current time
    historical_pods_now = get_historical_pod_count_at_time(historical_data, 0)
    if historical_pods_now == 0:
        historical_pods_now = 1  # Avoid division by zero
        
    # Get historical pod count 10 minutes ahead (negative because we're looking ahead)
    historical_pods_ahead = get_historical_pod_count_at_time(historical_data, -prediction_window_minutes)
    thread_logger.info(f"Historical data retrieved - historical pods now: {historical_pods_now}, historical pods ahead: {historical_pods_ahead}")

    # Calculate required pods using the formula
    ratio = current_pods / historical_pods_now
    thread_logger.info(f"Ratio calculated: {ratio} (current pods: {current_pods}, historical pods now: {historical_pods_now})")
    required_pods = ratio * historical_pods_ahead
    
    # Ensure it's within limits and an integer
    required_pods = min(int(required_pods), max_replicas)
    required_pods = max(required_pods, 1)
    thread_logger.info(f"Required pods calculated: {required_pods}")
    
    return required_pods

def update_hpa(namespace, hpa_name, min_replicas):
    try: 
        hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(hpa_name, namespace)
         
        hpa.spec.min_replicas = min_replicas
        autoscaling_api.patch_namespaced_horizontal_pod_autoscaler(
            name=hpa_name, 
            namespace=namespace, 
            body={"spec": {"minReplicas": min_replicas}}
        )
        thread_logger.info(f"Updated HPA {hpa_name} minReplicas to {min_replicas}")
        return True
    except kubernetes.client.exceptions.ApiException as e: 
        thread_logger.exception(f"Failed to update HPA {hpa_name}: {e}")
        return False

def update_status(namespace, name, status_data):
    try:
        # Format the lastUpdated timestamp in a more readable format
        if "lastUpdated" in status_data:
            status_data["lastUpdated"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        custom_api.patch_namespaced_custom_object_status(
            GROUP, VERSION, namespace, PLURAL, name, 
            {"status": status_data}
        )
    except kubernetes.client.exceptions.ApiException as e:
        thread_logger.exception(f"Failed to update status: {e}")

# Dictionary to track running timers
timers = {}

@kopf.on.create('scaler.cs.usc.edu', 'v1', 'predictiveautoscalers')
def create_fn(spec, name, namespace, logger, **kwargs):
    logger.info(f"Creating PredictiveAutoscaler {name} in namespace {namespace}")
    
    # Extract configuration
    target_deployment = spec['targetDeployment']
    target_hpa = spec['targetHPA']
    max_replicas = spec['maxReplicas']
    update_interval = spec.get('updateInterval', 5)  # default 5 minutes
    
    # Initialize historical data file if it doesn't exist
    historical_data = load_historical_data(namespace, name)
    current_pods = get_current_pod_count(namespace, target_deployment)
    
    if current_pods > 0:
        # Add initial entry with HH:MM format
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = f"{now.hour:02d}:{(now.minute // 5) * 5:02d}"
        historical_data["data"].append({"timestamp": timestamp, "podCount": current_pods})
        save_historical_data(historical_data, namespace, name)
        logger.info(f"Initialized historical data with timestamp {timestamp} and pod count {current_pods}")
    
    # Setup recurring job
    def recurring_update():
        if not check_if_cr_exists(namespace, name):
            thread_logger.info(f"PredictiveAutoscaler {name} no longer exists, stopping timer")
            if f"{namespace}_{name}" in timers:
                timers[f"{namespace}_{name}"].cancel()
                del timers[f"{namespace}_{name}"]
            return
        
        try:
            # Reload CR to get latest configuration
            cr = custom_api.get_namespaced_custom_object(
                GROUP, VERSION, namespace, PLURAL, name
            )
            spec = cr.get('spec', {})
            
            # Extract latest configuration
            target_deployment = spec['targetDeployment']
            target_hpa = spec['targetHPA']
            max_replicas = spec['maxReplicas']
            historical_weight = spec.get('historicalWeight', 0.7)
            current_weight = spec.get('currentWeight', 0.3)
            history_retention_days = spec.get('historyRetentionDays', 7)
            prediction_window_minutes = spec.get('predictionWindowMinutes', 10)
            update_interval = spec.get('updateInterval', 5)
            
            # Get current pod count
            current_pods = get_current_pod_count(namespace, target_deployment)
            thread_logger.info(f"Current pod count for {target_deployment}: {current_pods}")
            
            # Load historical data
            historical_data = load_historical_data(namespace, name)
            
            if current_pods > 0:
                # Calculate required pods
                required_pods = calculate_required_pods(
                    current_pods, historical_data, max_replicas, prediction_window_minutes)
                thread_logger.info(f"Required pods for next {prediction_window_minutes} minutes: {required_pods}")
                
                # Update HPA
                success = update_hpa(namespace, target_hpa, required_pods)
                
                # Update historical data
                updated_data = update_historical_data(
                    historical_data, current_pods, historical_weight, current_weight)
                updated_data = prune_old_data(updated_data, history_retention_days)
                save_historical_data(updated_data, namespace, name)
                thread_logger.info(f"Updated historical data to {updated_data}")
                
                # Update status
                status_data = {
                    "lastUpdated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "currentPrediction": required_pods
                }
                update_status(namespace, name, status_data)
            else:
                thread_logger.warning(f"Deployment {target_deployment} has 0 pods, skipping update")
                # Update status
                status_data = {
                    "lastUpdated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "lastError": "Deployment has 0 pods"
                }
                update_status(namespace, name, status_data)
            
            # Schedule next run
            timer = threading.Timer(update_interval * 60, recurring_update)
            timer.daemon = True
            timers[f"{namespace}_{name}"] = timer
            timer.start()
            
        except Exception as e:
            thread_logger.exception(f"Error in recurring update: {e}")
            # Update status with error
            status_data = {
                "lastUpdated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "lastError": str(e)
            }
            update_status(namespace, name, status_data)
            
            # Still schedule next run despite error
            timer = threading.Timer(update_interval * 60, recurring_update)
            timer.daemon = True
            timers[f"{namespace}_{name}"] = timer
            timer.start()
    
    # Start the first timer
    timer = threading.Timer(update_interval * 60, recurring_update)
    timer.daemon = True
    timers[f"{namespace}_{name}"] = timer
    timer.start()
    
    return {'autoscalerStarted': True}

def check_if_cr_exists(namespace, name):
    try:
        custom_api.get_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name
        )
        return True
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 404:
            return False
        raise

@kopf.on.delete('scaler.cs.usc.edu', 'v1', 'predictiveautoscalers')
def delete_fn(spec, name, namespace, logger, **kwargs):
    logger.info(f"Deleting PredictiveAutoscaler {name} in namespace {namespace}")
    
    # Cancel timer if running
    if f"{namespace}_{name}" in timers:
        timers[f"{namespace}_{name}"].cancel()
        del timers[f"{namespace}_{name}"]
    
    return {'autoscalerStopped': True}

@kopf.on.update('scaler.cs.usc.edu', 'v1', 'predictiveautoscalers')
def update_fn(spec, old, name, namespace, logger, **kwargs):
    logger.info(f"Updating PredictiveAutoscaler {name} in namespace {namespace}")
    if f"{namespace}_{name}" in timers:
        old_timer = timers[f"{namespace}_{name}"]
        old_timer.cancel()
        
        # Schedule immediate update
        def immediate_update():
            # Cancel existing timer first
            if f"{namespace}_{name}" in timers:
                timers[f"{namespace}_{name}"].cancel()
            
            # Get handler function from create_fn
            create_fn.__wrapped__(spec=spec, name=name, namespace=namespace, logger=logger)
        
        # Start immediate update
        timer = threading.Timer(0, immediate_update)
        timer.daemon = True
        timers[f"{namespace}_{name}"] = timer
        timer.start()
    
    return {'autoscalerUpdated': True}

if __name__ == "__main__":
    kopf.run() 