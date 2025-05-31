# Predictive Autoscaler for Kubernetes

A Kubernetes controller that implements predictive autoscaling based on historical workload patterns. This controller extends Kubernetes' Horizontal Pod Autoscaler (HPA) by incorporating predictive analytics to anticipate scaling needs before they occur.

## Features

- Predictive scaling based on historical workload patterns
- Configurable prediction intervals and algorithms
- Persistent storage of historical data
- Seamless integration with existing Kubernetes HPA
- Support for custom metrics and scaling rules

## Architecture

The controller:
1. Monitors the creation, update, and deletion of PredictiveAutoscaler custom resources
2. On a schedule (every 5 minutes by default), collects current pod counts
3. Calculates required pods using the predictive formula
4. Updates the target HPA's minReplicas
5. Updates the historical data with the weighted formula
6. Stores this data persistently for future predictions

## Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl configured to communicate with your cluster
- Docker installed for building the controller image
- Access to a container registry (Docker Hub, GCR, etc.)

## Quick Start

### 1. Build and Push the Controller Image

```bash
# Navigate to the controller directory
cd controller

# Build the Docker image
docker build -t predictive-autoscaler:latest .

# Tag and push to your container registry
docker tag predictive-autoscaler:latest <your-registry>/predictive-autoscaler:latest
docker push <your-registry>/predictive-autoscaler:latest
```

### 2. Deploy the Controller

```bash
# Apply the CRD
kubectl apply -f crd/predictive-autoscaler-crd.yaml

# Apply RBAC permissions
kubectl apply -f deploy/rbac.yaml

# Update the controller deployment with your image
kubectl apply -f deploy/controller-deployment.yaml

# Verify the controller is running
kubectl get pods -l app=predictive-autoscaler-controller
```

### 3. Create a PredictiveAutoscaler Instance

```bash
# Apply the example configuration
kubectl apply -f deploy/predictive-autoscaler-instance.yaml

# Verify the resource was created
kubectl get predictiveautoscalers
# or using the short name
kubectl get pa
```

## Configuration

### PredictiveAutoscaler Custom Resource

The controller is configured using a custom resource. Here's an example configuration:

```yaml
apiVersion: autoscaling.example.com/v1
kind: PredictiveAutoscaler
metadata:
  name: example-predictor
spec:
  targetHPA: example-hpa
  namespace: default
  predictionInterval: 5m
  historyRetention: 24h
  algorithm:
    type: weighted
    weight: 0.7
```

### Parameters

- `targetHPA`: Name of the HPA to control
- `namespace`: Kubernetes namespace
- `predictionInterval`: How often to make predictions
- `historyRetention`: How long to keep historical data
- `algorithm`: Prediction algorithm configuration

## Testing

### Using Sample Data

The repository includes sample traffic data for testing:

```bash
# Get the controller pod name
POD_NAME=$(kubectl get pods -l app=predictive-autoscaler-controller -o jsonpath="{.items[0].metadata.name}")

# Copy sample data
kubectl cp test-data/realistic-traffic.json $POD_NAME:/data/default_simpleweb-predictor_history.json
```

### Automated Testing

Use the provided test script:

```bash
cd testing
./test_with_traffic_1_interval.sh
```

## Monitoring

### View Controller Logs

```bash
kubectl logs -f deployment/predictive-autoscaler-controller
```

### Check HPA Status

```bash
kubectl get hpa
```

### View Historical Data

```bash
kubectl get predictiveautoscalers
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kubernetes HPA team for the base autoscaling functionality
- The open-source community for various tools and libraries used in this project
