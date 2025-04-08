#!/bin/bash
METRICS_FILE="metrics.csv"
NODE_FILE="nodes.csv"

# Initialize files
echo "timestamp,pod_count,hpa_desired,hpa_current,nodes_count" > $METRICS_FILE
echo "timestamp,node_name,creation_time" > $NODE_FILE

while true; do
    # Cluster metrics
    timestamp=$(date +%s)
    pod_count=$(kubectl get pods -l app=simpleweb -o json | jq '.items | length')
    hpa_status=$(kubectl get hpa simpleweb-hpa -o json | jq -c '[.status.desiredReplicas, .status.currentReplicas]')
    nodes_count=$(kubectl get nodes --no-headers | wc -l)
    
    # Append to metrics
    echo "$timestamp,$pod_count,${hpa_status//[\[\]]/},$nodes_count" >> $METRICS_FILE
    
    # Node creation times (for cost tracking)
    kubectl get nodes -o json | jq -r '.items[] | [.metadata.creationTimestamp, .metadata.name] | @csv' | \
    while read -r line; do
        echo "$timestamp,$line" >> $NODE_FILE
    done
    
    echo "Sleeping 5"
    sleep 5
done