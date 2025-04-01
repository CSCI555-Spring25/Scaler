# Get started

cd Scaler/webserver
docker run --rm -it -p 80:80 yeasy/simple-web:latest

cd ../testing

# Terminal 1: Port-forwarding
kubectl port-forward svc/simpleweb 30080:30080 &

# Terminal 2: Load tester
python load_tester.py

# Terminal 3: Proactive scaler
python proactive_scaler.py

# Terminal 4: Metrics collector
```bash
chmod +x metrics_collector.sh
./metrics_collector.sh
```