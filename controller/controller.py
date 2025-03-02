#!/usr/bin/env python3
import kopf
import kubernetes
import os
import json
import datetime
import time
import threading

# Load Kubernetes configuration
kubernetes.config.load_incluster_config()

# Initialize clients
api_client = kubernetes.client.ApiClient()
apps_api = kubernetes.client.AppsV1Api(api_client)
autoscaling_api = kubernetes.client.AutoscalingV2Api(api_client)
custom_api = kubernetes.client.CustomObjectsApi(api_client)

# Constants
DATA_DIR = "/data"
GROUP = "scaler.cs.columbia.edu"
VERSION = "v1"
PLURAL = "predictiveautoscalers"

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
            kopf.info(f"Deployment {deployment_name} not found in namespace {namespace}")
            return 0
        raise

def get_historical_pod_count_at_time(data, time_offset_minutes=0):
    now = datetime.datetime.now(datetime.timezone.utc)
    target_time = now - datetime.timedelta(minutes=time_offset_minutes)
    
    # Find the closest historical data point
    target_hour = target_time.hour
    target_minute = (target_time.minute // 5) * 5  # Round to nearest 5 minutes
    
    for entry in data["data"]:
        timestamp = datetime.datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
        if timestamp.hour == target_hour and timestamp.minute == target_minute:
            return entry["podCount"]
    
    # Default if not found
    return 1

def update_historical_data(data, current_pods, historical_weight=0.7, current_weight=0.3):
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    
    # Find if we have an entry for this timestamp
    for entry in data["data"]:
        if entry["timestamp"] == timestamp:
            historical_count = entry["podCount"]
            # Update using the given formula
            entry["podCount"] = int(historical_weight * historical_count + current_weight * current_pods)
            break
    else:
        # No entry found, create new one
        data["data"].append({"timestamp": timestamp, "podCount": current_pods})
    
    # Sort by timestamp
    data["data"] = sorted(data["data"], key=lambda x: x["timestamp"])
    
    return data

def prune_old_data(data, retention_days=7):
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=retention_days)
    cutoff_timestamp = cutoff_date.isoformat().replace("+00:00", "Z")
    
    data["data"] = [entry for entry in data["data"] if entry["timestamp"] >= cutoff_timestamp]
    return data

def calculate_required_pods(current_pods, historical_data, max_replicas, prediction_window_minutes=10):
    # Get historical pod count at current time
    historical_pods_now = get_historical_pod_count_at_time(historical_data, 0)
    if historical_pods_now == 0:
        historical_pods_now = 1  # Avoid division by zero
        
    # Get historical pod count 10 minutes ahead (negative because we're looking ahead)
    historical_pods_ahead = get_historical_pod_count_at_time(historical_data, -prediction_window_minutes)
    
    # Calculate required pods using the formula
    ratio = current_pods / historical_pods_now
    required_pods = ratio * historical_pods_ahead
    
    # Ensure it's within limits and an integer
    required_pods = min(int(required_pods), max_replicas)
    required_pods = max(required_pods, 1)
    
    return required_pods

def update_hpa(namespace, hpa_name, required_pods):
    try:
        # Get current HPA
        hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(hpa_name, namespace)
        
        # Update the minReplicas to our calculated value
        hpa.spec.min_replicas = required_pods
        
        # Update the HPA
        autoscaling_api.replace_namespaced_horizontal_pod_autoscaler(hpa_name, namespace, hpa)
        kopf.info(f"Updated HPA {hpa_name} minReplicas to {required_pods}")
        return True
    except kubernetes.client.exceptions.ApiException as e:
        kopf.exception(f"Failed to update HPA: {e}")
        return False

def update_status(namespace, name, status_data):
    try:
        custom_api.patch_namespaced_custom_object_status(
            GROUP, VERSION, namespace, PLURAL, name, 
            {"status": status_data}
        )
    except kubernetes.client.exceptions.ApiException as e:
        kopf.exception(f"Failed to update status: {e}")

# Dictionary to track running timers
timers = {}

@kopf.on.create('scaler.cs.columbia.edu', 'v1', 'predictiveautoscalers')
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
        # Add initial entry
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
        historical_data["data"].append({"timestamp": timestamp, "podCount": current_pods})
        save_historical_data(historical_data, namespace, name)
    
    # Setup recurring job
    def recurring_update():
        if not check_if_cr_exists(namespace, name):
            logger.info(f"PredictiveAutoscaler {name} no longer exists, stopping timer")
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
            logger.info(f"Current pod count for {target_deployment}: {current_pods}")
            
            # Load historical data
            historical_data = load_historical_data(namespace, name)
            
            if current_pods > 0:
                # Calculate required pods
                required_pods = calculate_required_pods(
                    current_pods, historical_data, max_replicas, prediction_window_minutes)
                logger.info(f"Required pods for next {prediction_window_minutes} minutes: {required_pods}")
                
                # Update HPA
                success = update_hpa(namespace, target_hpa, required_pods)
                
                # Update historical data
                updated_data = update_historical_data(
                    historical_data, current_pods, historical_weight, current_weight)
                updated_data = prune_old_data(updated_data, history_retention_days)
                save_historical_data(updated_data, namespace, name)
                logger.info("Updated historical data")
                
                # Update status
                status_data = {
                    "lastUpdated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "currentPrediction": required_pods
                }
                update_status(namespace, name, status_data)
            else:
                logger.warning(f"Deployment {target_deployment} has 0 pods, skipping update")
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
            logger.exception(f"Error in recurring update: {e}")
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

@kopf.on.delete('scaler.cs.columbia.edu', 'v1', 'predictiveautoscalers')
def delete_fn(spec, name, namespace, logger, **kwargs):
    logger.info(f"Deleting PredictiveAutoscaler {name} in namespace {namespace}")
    
    # Cancel timer if running
    if f"{namespace}_{name}" in timers:
        timers[f"{namespace}_{name}"].cancel()
        del timers[f"{namespace}_{name}"]
    
    return {'autoscalerStopped': True}

@kopf.on.update('scaler.cs.columbia.edu', 'v1', 'predictiveautoscalers')
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