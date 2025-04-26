#!/bin/bash
METRICS_FILE="metrics.csv"
NODE_FILE="nodes.csv"
POD_FILE="pods.csv"

# Initialize files
echo "timestamp,pod_count,hpa_desired,hpa_current,nodes_count" > "$METRICS_FILE"
echo "timestamp,node_name,creation_time"           > "$NODE_FILE"
echo "timestamp,pod_name,creation_time"            > "$POD_FILE"

while true; do
    # Readable timestamp: month-day, hour:minute:second
    timestamp=$(date +'%m-%d,%H:%M:%S')

    # Cluster metrics
    pod_count=$(kubectl get pods -l app=simpleweb -o json | jq '.items | length')
    read -r hpa_desired hpa_current <<< "$(kubectl get hpa simpleweb-hpa -o json | jq -r '[.status.desiredReplicas, .status.currentReplicas] | @tsv')"
    nodes_count=$(kubectl get nodes --no-headers | wc -l)
    
    # Append to metrics
    echo "$timestamp,$pod_count,$hpa_desired,$hpa_current,$nodes_count" >> "$METRICS_FILE"
    
    # Node creation times
    kubectl get nodes -o json | jq -r '.items[] | [.metadata.name, .metadata.creationTimestamp] | @csv' | \
    while IFS=, read -r node_name creation_time; do
        echo "$timestamp,$node_name,$creation_time" >> "$NODE_FILE"
    done

    # Pod creation times
    kubectl get pods -l app=simpleweb -o json | jq -r '.items[] | [.metadata.name, .metadata.creationTimestamp] | @csv' | \
    while IFS=, read -r pod_name creation_time; do
        echo "$timestamp,$pod_name,$creation_time" >> "$POD_FILE"
    done
    
    echo "Sleeping 5"
    sleep 5
done
