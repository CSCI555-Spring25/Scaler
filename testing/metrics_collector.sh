#!/bin/bash
METRICS_FILE="metrics.csv"
NODE_FILE="nodes.csv"
RESOURCE_FILE="resources.csv"

# Initialize files
echo "timestamp,pod_count,hpa_desired,hpa_current,nodes_count,cpu_utilization" > $METRICS_FILE
echo "timestamp,node_name,creation_time,status" > $NODE_FILE
echo "timestamp,pod_name,cpu_request,cpu_limit,memory_request,memory_limit,cpu_usage,memory_usage" > $RESOURCE_FILE

collect_metrics() {
    # Cluster metrics
    timestamp=$(date +%s)
    pod_count=$(kubectl get pods -l app=simpleweb -o json | jq '.items | length')
    hpa_status=$(kubectl get hpa simpleweb-hpa -o json 2>/dev/null | jq -c '[.status.desiredReplicas, .status.currentReplicas]')
    nodes_count=$(kubectl get nodes --no-headers | wc -l)
    
    # Get CPU utilization from HPA if available
    cpu_util=$(kubectl get hpa simpleweb-hpa -o json 2>/dev/null | \
               jq -r '.status.currentMetrics[] | select(.type=="Resource" and .resource.name=="cpu") | .resource.current.averageUtilization // "NA"')
    
    # Append to metrics
    echo "$timestamp,$pod_count,${hpa_status//[\[\]]/},$nodes_count,$cpu_util" >> $METRICS_FILE
    
    # Node creation times and status (for cost tracking)
    kubectl get nodes -o json | jq -r '.items[] | [.metadata.creationTimestamp, .metadata.name, .status.conditions[-1].type] | @csv' | \
    while read -r line; do
        echo "$timestamp,$line" >> $NODE_FILE
    done
    
    # Resource usage per pod
    kubectl get pods -l app=simpleweb -o json | jq -r '.items[] | .metadata.name' | \
    while read -r pod_name; do
        # Get resource requests and limits
        requests=$(kubectl get pod $pod_name -o json | \
                 jq -r '.spec.containers[0].resources.requests | (.cpu // "NA") + "," + (.memory // "NA")')
        limits=$(kubectl get pod $pod_name -o json | \
               jq -r '.spec.containers[0].resources.limits | (.cpu // "NA") + "," + (.memory // "NA")')
        
        # Get current usage (requires metrics-server)
        usage=$(kubectl top pod $pod_name --no-headers 2>/dev/null | awk '{print $2","$3}')
        
        if [ -z "$usage" ]; then
            usage="NA,NA"
        fi
        
        echo "$timestamp,$pod_name,$requests,$limits,$usage" >> $RESOURCE_FILE
    done
    
    echo "$(date): Collected metrics"
}

# Main loop
while true; do
    collect_metrics
    sleep 5
done