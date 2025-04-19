# Get started

cd Scaler/webserver
docker run --rm -it -p 80:80 yeasy/simple-web:latest

cd ../testing

# Terminal 1: Port-forwarding
kubectl port-forward svc/simpleweb 30080:30080 &

# Terminal 2: Load tester
python load_tester.py

# Terminal 3: Proactive scaler
kubectl get hpa simpleweb-hpa --watch

# Terminal 4: Metrics collector
```bash
chmod +x metrics_collector.sh
./metrics_collector.sh
```


nohup python3 load_tester.py > output.log 2>&1 &

# Testing Instructions

## Testing with traffic_1_interval.json

1. Apply the CRD and deploy the controller:
   ```
   kubectl apply -f Scaler/crd/predictive-autoscaler-crd.yaml
   kubectl apply -f Scaler/deploy/rbac.yaml
   kubectl apply -f Scaler/deploy/controller-deployment.yaml
   ```

2. Copy the traffic_1_interval.json file to the controller pod:
   ```
   # Get the pod name
   POD_NAME=$(kubectl get pods -l app=predictive-autoscaler-controller -o jsonpath="{.items[0].metadata.name}")
   
   # Create the data directory if needed
   kubectl exec -it $POD_NAME -- mkdir -p /data
   
   # Copy the traffic_1_interval.json file to the pod
   kubectl cp CSCI555/Scaler/testing/json/traffic_1_interval.json $POD_NAME:/data/traffic_1_interval.json
   ```

3. Deploy the sample application and HPA:
   ```
   kubectl apply -f Scaler/deploy/simpleweb-deployment.yaml
   kubectl apply -f Scaler/deploy/simpleweb-hpa.yaml
   ```

4. Create the predictive autoscaler instance:
   ```
   kubectl apply -f Scaler/deploy/predictive-autoscaler-instance.yaml
   ```

5. Monitor logs and HPA updates:
   ```
   kubectl logs -f $POD_NAME
   kubectl get hpa simpleweb-hpa -w
   ```


