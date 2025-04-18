#!/usr/bin/env python3
import kopf
import kubernetes
import os
import json
import datetime
import time
import threading
import logging
import subprocess
import pytz

# K8s config 
kubernetes.config.load_incluster_config()

# Clients
api_client = kubernetes.client.ApiClient()
apps_api = kubernetes.client.AppsV1Api(api_client)
autoscaling_api = kubernetes.client.AutoscalingV2Api(api_client)
custom_api = kubernetes.client.CustomObjectsApi(api_client)

# Constants
DATA_DIR = "/data"
GROUP = "scaler.cs.usc.edu"
VERSION = "v1"
PLURAL = "predictiveautoscalers"
TIMEZONE = pytz.timezone('America/Chicago')

# Configure logging
logger = logging.getLogger(__name__)
thread_logger = logging.getLogger(f"{__name__}.thread")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def ensure_data_dir():
    logger.info(f"Ensuring data directory exists: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)

def get_history_file_path(namespace=None, name=None):
    file_path = os.path.join(DATA_DIR, "traffic_1_interval.json")
    if not os.path.exists(file_path):
        logger.warning(f"History file {file_path} does not exist, returning empty data")
        file_path = os.path.join(DATA_DIR, f"{namespace}_{name}_history.json")
    logger.debug(f"Using history file path: {file_path}")
    return file_path

def load_historical_data(namespace=None, name=None):
    file_path = get_history_file_path(namespace, name)
    logger.info(f"Loading historical data from {file_path}")
    if not os.path.exists(file_path):
        logger.warning(f"History file {file_path} does not exist, returning empty data")
        return {"data": []}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # If the data is already in our format, return it
            if isinstance(data, dict) and "data" in data:
                logger.info(f"Loaded {len(data['data'])} historical data entries")
                return data
            # If it's a list (like traffic_1_interval.json), convert it
            elif isinstance(data, list):
                logger.info(f"Loaded {len(data)} historical data entries (list format), converting to dict")
                return {"data": data}
            else:
                logger.warning(f"Unexpected data format in {file_path}, returning empty data")
                return {"data": []}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {file_path}: {e}")
        return {"data": []}

def save_historical_data(data, namespace=None, name=None):
    ensure_data_dir()
    file_path = get_history_file_path(namespace, name)
    logger.info(f"Saving historical data to {file_path}")
    
    # If data is a dict with "data" key, save it as is
    if isinstance(data, dict) and "data" in data:
        logger.info(f"Saving {len(data['data'])} historical data entries")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    # If data is a list, convert to our format
    elif isinstance(data, list):
        logger.info(f"Saving {len(data)} historical data entries (list format)")
        with open(file_path, 'w') as f:
            json.dump({"data": data}, f, indent=2)
    logger.debug(f"Historical data saved successfully to {file_path}")

def get_current_pod_count(namespace, deployment_name):
    logger.info(f"Getting current pod count for deployment {deployment_name} in namespace {namespace}")
    try:
        deployment = apps_api.read_namespaced_deployment(deployment_name, namespace)
        pod_count = deployment.status.replicas or 0
        logger.info(f"Current pod count for {deployment_name}: {pod_count}")
        return pod_count
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 404:
            logger.info(f"Deployment {deployment_name} not found in namespace {namespace}")
            return 0
        logger.error(f"API error when getting pod count: {e}")
        raise

def get_node_time():
    try:
        now = datetime.datetime.now(TIMEZONE)
        time_str = now.strftime('%H:%M')
        logger.debug(f"Node time (America/Chicago): {time_str}")
        return time_str
    except Exception as e:
        logger.error(f"Error getting node time: {e}")
        # Fallback to system time
        now = datetime.datetime.now()
        return f"{now.hour:02d}:{now.minute:02d}"

def get_historical_pod_count_at_time(data, time_offset_minutes=0):
    # Get current time from node
    now_str = get_node_time()
    now_hour, now_minute = map(int, now_str.split(':'))
    
    # Calculate target time
    total_minutes = now_hour * 60 + now_minute - time_offset_minutes
    target_hour = (total_minutes // 60) % 24
    target_minute = total_minutes % 60
    
    # Target timestamp in HH:MM format
    target_timestamp = f"{target_hour:02d}:{target_minute:02d}"
    logger.info(f"Looking for historical data at {target_timestamp} (offset: {time_offset_minutes} minutes from {now_str})")
    
    # Ensure we're working with the data list
    data_list = data["data"] if isinstance(data, dict) and "data" in data else data
    logger.debug(f"Working with {len(data_list)} historical data entries")
    
    # Exact match check
    for entry in data_list:
        if entry["timestamp"] == target_timestamp:
            logger.info(f"Found exact match for {target_timestamp} with pod count {entry['podCount']}")
            return entry["podCount"]
    
    # If exact match not found, find closest timestamp that's earlier 
    target_minutes = target_hour * 60 + target_minute
    logger.debug(f"No exact match found, looking for closest earlier timestamp to {target_timestamp} ({target_minutes} minutes)")
    
    closest_entry = None
    closest_diff = float('inf')
    
    for entry in data_list: 
        h, m = map(int, entry["timestamp"].split(':'))
        entry_minutes = h * 60 + m
        if entry_minutes <= target_minutes:
            diff = target_minutes - entry_minutes
            if diff < closest_diff:
                closest_diff = diff
                closest_entry = entry
                logger.debug(f"Found closer timestamp {entry['timestamp']} ({diff} minutes difference)")
    
    if closest_entry:
        logger.info(f"Found closest earlier timestamp {closest_entry['timestamp']} with pod count {closest_entry['podCount']} ({closest_diff} minutes difference)")
        return closest_entry["podCount"]
    
    logger.info(f"No historical data found for {target_timestamp}, using default pod count 1")
    return 1

def update_historical_data(data, current_pods, historical_weight=0.7, current_weight=0.3):
    # Get current time from node
    timestamp = get_node_time()
    logger.info(f"Updating historical data for timestamp {timestamp} with current pods {current_pods} (weights: historical={historical_weight}, current={current_weight})")
    
    # Ensure we're working with the data list
    data_list = data["data"] if isinstance(data, dict) and "data" in data else data
    logger.debug(f"Working with {len(data_list)} historical data entries")
    
    for entry in data_list:
        if entry["timestamp"] == timestamp:
            historical_count = entry["podCount"]
            new_count = int(historical_weight * historical_count + current_weight * current_pods)
            logger.info(f"Updated historical data for timestamp {timestamp} from {entry['podCount']} to {new_count} (historical: {historical_count}, current: {current_pods})")
            entry["podCount"] = new_count
            break
    else:
        logger.info(f"Created new historical data entry for timestamp {timestamp} with pod count {current_pods}")
        data_list.append({"timestamp": timestamp, "podCount": current_pods})
     
    def timestamp_to_minutes(ts):
        h, m = map(int, ts.split(':'))
        return h * 60 + m
    
    logger.debug(f"Sorting {len(data_list)} historical data entries")
    data_list = sorted(data_list, key=lambda x: timestamp_to_minutes(x["timestamp"]))
    
    # Return in our standard format
    if isinstance(data, dict) and "data" in data:
        data["data"] = data_list
        logger.debug(f"Returning updated data dictionary with {len(data_list)} entries")
        return data
    else:
        logger.debug(f"Returning new data dictionary with {len(data_list)} entries")
        return {"data": data_list}

def prune_old_data(data, retention_days=7): 
    #Prune data older than retention_days
    logger.info(f"Historical data has {len(data['data'])} entries (retention: {retention_days} days)")
    # TODO: Implement actual pruning logic
    return data

def calculate_required_pods(current_pods, historical_data, max_replicas, prediction_window_minutes=10):
    logger.info(f"Calculating required pods (current: {current_pods}, max: {max_replicas}, window: {prediction_window_minutes} minutes)")
    historical_pods_now = get_historical_pod_count_at_time(historical_data, 0)
    
    if historical_pods_now == 0: # Divide by zero error fix
        logger.warning("Historical pods now is 0, using 1 to avoid division by zero")
        historical_pods_now = 1  
         
    historical_pods_ahead = get_historical_pod_count_at_time(historical_data, -prediction_window_minutes) # Negative for future
    thread_logger.info(f"Historical data retrieved - historical pods now: {historical_pods_now}, historical pods ahead: {historical_pods_ahead}")
 
    ratio = current_pods / historical_pods_now
    thread_logger.info(f"Ratio calculated: {ratio} (current pods: {current_pods}, historical pods now: {historical_pods_now})")
    required_pods = ratio * historical_pods_ahead
    logger.debug(f"Raw required pods calculation: {required_pods} = {ratio} * {historical_pods_ahead}")
     
    required_pods = min(int(required_pods), max_replicas)
    required_pods = max(required_pods, 1)
    thread_logger.info(f"Required pods calculated: {required_pods} (bounded between 1 and {max_replicas})")
    
    return required_pods

def update_hpa(namespace, hpa_name, min_replicas):
    logger.info(f"Updating HPA {hpa_name} in namespace {namespace} to minReplicas={min_replicas}")
    try: 
        hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(hpa_name, namespace)
        logger.debug(f"Current HPA {hpa_name} settings: minReplicas={hpa.spec.min_replicas}, maxReplicas={hpa.spec.max_replicas}")
         
        hpa.spec.min_replicas = min_replicas
        autoscaling_api.patch_namespaced_horizontal_pod_autoscaler(
            name=hpa_name, 
            namespace=namespace, 
            body={"spec": {"minReplicas": min_replicas}}
        )
        thread_logger.info(f"Successfully updated HPA {hpa_name} minReplicas to {min_replicas}")
        return True
    except kubernetes.client.exceptions.ApiException as e: 
        thread_logger.exception(f"Failed to update HPA {hpa_name}: {e}")
        return False

def update_status(namespace, name, status_data):
    logger.info(f"Updating status for PredictiveAutoscaler {name} in namespace {namespace}")
    try: 
        if "lastUpdated" in status_data:
            # Use Chicago timezone for status updates
            now = datetime.datetime.now(TIMEZONE)
            status_data["lastUpdated"] = now.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        logger.debug(f"Status data: {status_data}")
        custom_api.patch_namespaced_custom_object_status(
            GROUP, VERSION, namespace, PLURAL, name, 
            {"status": status_data}
        )
        logger.info(f"Status updated successfully for {name}")
    except kubernetes.client.exceptions.ApiException as e:
        thread_logger.exception(f"Failed to update status for {name}: {e}")
 
timers = {}

@kopf.on.create('scaler.cs.usc.edu', 'v1', 'predictiveautoscalers')
def create_fn(spec, name, namespace, logger, **kwargs):
    logger.info(f"Creating PredictiveAutoscaler {name} in namespace {namespace}")
    logger.debug(f"Spec: {spec}")
    
    # Configuration
    target_deployment = spec['targetDeployment']
    target_hpa = spec['targetHPA']
    max_replicas = spec['maxReplicas']
    update_interval = spec.get('updateInterval', 5)  # default 5 minutes
    
    logger.info(f"Configuration: target_deployment={target_deployment}, target_hpa={target_hpa}, max_replicas={max_replicas}, update_interval={update_interval}")
    
    historical_data = load_historical_data(namespace, name)
    current_pods = get_current_pod_count(namespace, target_deployment)
    logger.info(f"Initial state: current_pods={current_pods}, historical_data_entries={len(historical_data['data'])}")
    
    if current_pods > 0: 
        timestamp = get_node_time()
        historical_data["data"].append({"timestamp": timestamp, "podCount": current_pods})
        save_historical_data(historical_data, namespace, name)
        logger.info(f"Initialized historical data with timestamp {timestamp} and pod count {current_pods}")
     
    def recurring_update():
        thread_logger.info(f"Running recurring update for {name} in namespace {namespace}")
        if not check_if_cr_exists(namespace, name):
            thread_logger.info(f"PredictiveAutoscaler {name} no longer exists, stopping timer")
            if f"{namespace}_{name}" in timers:
                timers[f"{namespace}_{name}"].cancel()
                del timers[f"{namespace}_{name}"]
            return
        
        try: 
            thread_logger.info(f"Fetching current CR for {name}")
            cr = custom_api.get_namespaced_custom_object(
                GROUP, VERSION, namespace, PLURAL, name
            )
            spec = cr.get('spec', {})
            thread_logger.debug(f"Current spec: {spec}")
             
            target_deployment = spec['targetDeployment']
            target_hpa = spec['targetHPA']
            max_replicas = spec['maxReplicas']
            historical_weight = spec.get('historicalWeight', 0.7)
            current_weight = spec.get('currentWeight', 0.3)
            history_retention_days = spec.get('historyRetentionDays', 7)
            prediction_window_minutes = spec.get('predictionWindowMinutes', 10)
            update_interval = spec.get('updateInterval', 5)
            
            thread_logger.info(f"Configuration: target_deployment={target_deployment}, target_hpa={target_hpa}, max_replicas={max_replicas}, historical_weight={historical_weight}, current_weight={current_weight}, history_retention_days={history_retention_days}, prediction_window_minutes={prediction_window_minutes}, update_interval={update_interval}")
             
            current_pods = get_current_pod_count(namespace, target_deployment)
            thread_logger.info(f"Current pod count for {target_deployment}: {current_pods}")
             
            historical_data = load_historical_data(namespace, name)
            thread_logger.info(f"Loaded {len(historical_data['data'])} historical data entries")
            
            if current_pods > 0: 
                thread_logger.info(f"Calculating required pods for next {prediction_window_minutes} minutes")
                required_pods = calculate_required_pods(
                    current_pods, historical_data, max_replicas, prediction_window_minutes)
                thread_logger.info(f"Required pods for next {prediction_window_minutes} minutes: {required_pods}")
                
                # Update HPA
                thread_logger.info(f"Updating HPA {target_hpa} to minReplicas={required_pods}")
                success = update_hpa(namespace, target_hpa, required_pods)
                thread_logger.info(f"HPA update {'succeeded' if success else 'failed'}")
                
                # Update historical data
                thread_logger.info(f"Updating historical data with current_pods={current_pods}")
                updated_data = update_historical_data(
                    historical_data, current_pods, historical_weight, current_weight)
                thread_logger.info(f"Pruning old data (retention: {history_retention_days} days)")
                updated_data = prune_old_data(updated_data, history_retention_days)
                thread_logger.info(f"Saving updated historical data ({len(updated_data['data'])} entries)")
                save_historical_data(updated_data, namespace, name)
                thread_logger.info(f"Historical data updated successfully")
                 
                status_data = {
                    "lastUpdated": "timestamp",  # Will be replaced in update_status
                    "currentPrediction": required_pods
                }
                thread_logger.info(f"Updating status with currentPrediction={required_pods}")
                update_status(namespace, name, status_data)
            else:
                thread_logger.warning(f"Deployment {target_deployment} has 0 pods, skipping update")
                status_data = {
                    "lastUpdated": "timestamp",  # Will be replaced in update_status
                    "lastError": "Deployment has 0 pods"
                }
                thread_logger.info(f"Updating status with error: Deployment has 0 pods")
                update_status(namespace, name, status_data)
            
            thread_logger.info(f"Scheduling next update in {update_interval} minutes")
            timer = threading.Timer(update_interval * 60, recurring_update)
            timer.daemon = True
            timers[f"{namespace}_{name}"] = timer
            timer.start()
            
        except Exception as e:
            thread_logger.exception(f"Error in recurring update: {e}") 
            status_data = {
                "lastUpdated": "timestamp",  # Will be replaced in update_status
                "lastError": str(e)
            }
            thread_logger.info(f"Updating status with error: {str(e)}")
            update_status(namespace, name, status_data)
            
            thread_logger.info(f"Scheduling next update in {update_interval} minutes despite error")
            timer = threading.Timer(update_interval * 60, recurring_update)
            timer.daemon = True
            timers[f"{namespace}_{name}"] = timer
            timer.start()
    
    logger.info(f"Scheduling first update in {update_interval} minutes")
    timer = threading.Timer(update_interval * 60, recurring_update)
    timer.daemon = True
    timers[f"{namespace}_{name}"] = timer
    timer.start()
    
    logger.info(f"PredictiveAutoscaler {name} created successfully")
    return {'autoscalerStarted': True}

def check_if_cr_exists(namespace, name):
    logger.debug(f"Checking if CR {name} exists in namespace {namespace}")
    try:
        custom_api.get_namespaced_custom_object(
            GROUP, VERSION, namespace, PLURAL, name
        )
        logger.debug(f"CR {name} exists in namespace {namespace}")
        return True
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 404:
            logger.debug(f"CR {name} does not exist in namespace {namespace}")
            return False
        logger.error(f"Error checking if CR exists: {e}")
        raise

@kopf.on.delete('scaler.cs.usc.edu', 'v1', 'predictiveautoscalers')
def delete_fn(spec, name, namespace, logger, **kwargs):
    logger.info(f"Deleting PredictiveAutoscaler {name} in namespace {namespace}")
     
    if f"{namespace}_{name}" in timers:
        logger.info(f"Cancelling timer for {name}")
        timers[f"{namespace}_{name}"].cancel()
        del timers[f"{namespace}_{name}"]
    
    logger.info(f"PredictiveAutoscaler {name} deleted successfully")
    return {'autoscalerStopped': True}

@kopf.on.update('scaler.cs.usc.edu', 'v1', 'predictiveautoscalers')
def update_fn(spec, old, name, namespace, logger, **kwargs):
    logger.info(f"Updating PredictiveAutoscaler {name} in namespace {namespace}")
    logger.debug(f"New spec: {spec}")
    
    if f"{namespace}_{name}" in timers:
        logger.info(f"Cancelling existing timer for {name}")
        old_timer = timers[f"{namespace}_{name}"]
        old_timer.cancel()
         
        def immediate_update(): 
            logger.info(f"Running immediate update for {name} after spec change")
            if f"{namespace}_{name}" in timers:
                timers[f"{namespace}_{name}"].cancel() 
            create_fn.__wrapped__(spec=spec, name=name, namespace=namespace, logger=logger)

        logger.info(f"Scheduling immediate update for {name}")
        timer = threading.Timer(0, immediate_update)
        timer.daemon = True
        timers[f"{namespace}_{name}"] = timer
        timer.start()
    
    logger.info(f"PredictiveAutoscaler {name} updated successfully")
    return {'autoscalerUpdated': True}

if __name__ == "__main__":
    logger.info("Starting Predictive Autoscaler controller")
    kopf.run() 