#!/bin/bash

# Exit on any error
set -e

echo "=== Installing Predictive Autoscaler with traffic_1_interval.json ==="

# Apply CRD and RBAC
echo "Applying CRD and RBAC..."
kubectl apply -f ../crd/predictive-autoscaler-crd.yaml
kubectl apply -f ../deploy/rbac.yaml

# Apply controller deployment
echo "Deploying controller..."
kubectl apply -f ../deploy/controller-deployment.yaml

# Wait for controller to be ready
echo "Waiting for controller to be ready..."
kubectl wait --for=condition=ready pod -l app=predictive-autoscaler-controller --timeout=120s

# Get pod name
POD_NAME=$(kubectl get pods -l app=predictive-autoscaler-controller -o jsonpath="{.items[0].metadata.name}")
echo "Controller pod: $POD_NAME"

# Create data directory and copy traffic data
echo "Copying traffic_1_interval.json to controller pod..."
kubectl exec -it $POD_NAME -- mkdir -p /data
kubectl cp json/traffic_1_interval.json $POD_NAME:/data/traffic_1_interval.json

# Deploy test application
echo "Deploying test application and HPA..."
kubectl apply -f ../deploy/simpleweb-deployment.yaml
kubectl apply -f ../deploy/simpleweb-hpa.yaml

# Apply predictive autoscaler
echo "Creating PredictiveAutoscaler instance..."
kubectl apply -f ../deploy/predictive-autoscaler-instance.yaml

echo "=== Setup complete ==="
echo ""
echo "To monitor HPA updates:"
echo "kubectl get hpa simpleweb-hpa -w"
echo ""
echo "To view controller logs:"
echo "kubectl logs -f $POD_NAME"
echo ""
echo "To view the PredictiveAutoscaler status:"
echo "kubectl get pa simpleweb-predictor -o yaml" 