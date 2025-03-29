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
`docker tag predictive-autoscaler:latest anirudhr120100/csci555-predictive-autoscaler:latest`

`docker push anirudhr120100/csci555-predictive-autoscaler:latest`

### Update the controller deployment with your image registry
Edit /Scaler/deploy/controller-deployment.yaml 

### Apply CRD first
`kubectl apply -f Scaler/crd/predictive-autoscaler-crd.yaml`

### Apply RBAC
`kubectl apply -f Scaler/deploy/rbac.yaml`

### Apply controller deployment
`kubectl apply -f Scaler/deploy/controller-deployment.yaml`

### Check if the controller is running
`kubectl get pods -l app=predictive-autoscaler-controller`

### Apply the PredictiveAutoscaler instance
`kubectl apply -f Scaler/deploy/predictive-autoscaler-instance.yaml`

### Check if the PredictiveAutoscaler custom resource was created 
`kubectl get predictiveautoscalers` 
or
`kubectl get pa` (using the short name)

### Test the controller
#### Create the directory in your persistent volume if needed
`kubectl exec -it predictive-autoscaler-controller-xxx -- mkdir -p /data`

#### Copy the sample data file to your controller pod
`kubectl cp Scaler/test-data/realistic-traffic.json predictive-autoscaler-controller-xxx:/data/default_simpleweb-predictor_history.json`

#### Check the logs to see if the controller is using the data correctly
`kubectl logs -f deployment/predictive-autoscaler-controller`

#### Check if the HPA is being updated
`kubectl get hpa`

#### Check if the historical data is being updated
`kubectl get predictiveautoscalers`
