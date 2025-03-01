# How It Works

The controller:
1. Monitors the creation, update, and deletion of PredictiveAutoscaler custom resources
2. On a schedule (every 5 minutes by default), collects current pod counts
3. Calculates required pods using the predictive formula
4. Updates the target HPA's minReplicas
5. Updates the historical data with the weighted formula
6. Stores this data persistently for future predictions

This implementation provides a clean separation of code and configuration, making it easier to develop, test, and maintain the controller.

## Deployment Instructions

### Build and Push the Controller Image

Navigate to the controller directory
cd CSCI555/Scaler/controller

Build the Docker image
`docker build -t predictive-autoscaler:latest .`

Tag and push to your container registry
`docker tag predictive-autoscaler:latest ${REGISTRY}/predictive-autoscaler:latest`

`docker push ${REGISTRY}/predictive-autoscaler:latest`

### Update the controller deployment with your image registry
Edit CSCI555/Scaler/deploy/controller-deployment.yaml and replace ${REGISTRY} with actual container registry.

### Apply CRD first
`kubectl apply -f CSCI555/Scaler/crd/predictive-autoscaler-crd.yaml`

### Apply RBAC
`kubectl apply -f CSCI555/Scaler/deploy/rbac.yaml`

### Apply controller deployment
`kubectl apply -f CSCI555/Scaler/deploy/controller-deployment.yaml`

### Apply the PredictiveAutoscaler instance
`kubectl apply -f CSCI555/Scaler/deploy/predictive-autoscaler-instance.yaml`

### Check if the controller is running
`kubectl get pods -l app=predictive-autoscaler-controller`

### Check if the PredictiveAutoscaler instance is running   
`kubectl get pods -l app=predictive-autoscaler-instance`


### Test the controller
#### Create the directory in your persistent volume if needed
`kubectl exec -it predictive-autoscaler-controller-xxx -- mkdir -p /data`

#### Copy the sample data file to your controller pod
`kubectl cp CSCI555/Scaler/test-data/sample-history.json predictive-autoscaler-controller-xxx:/data/default_simpleweb-predictor_history.json`

#### Check the logs to see if the controller is using the data correctly
`kubectl logs -f deployment/predictive-autoscaler-controller`