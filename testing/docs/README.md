# Kubernetes Proactive Scaling System

This repository contains a set of tools for implementing and testing proactive scaling in Kubernetes. The system uses historical data to predict and scale resources before they're needed, rather than reacting to current load.

## Components

### 2. Load Tester (`load_tester.py`)
A realistic load generation tool that:
- Simulates daily traffic patterns using a Gaussian distribution
- Peaks at configured times (default: 6 PM)
- Records historical load data
- Uses Apache Bench (ab) for load testing
- Adds random noise to make patterns more realistic

### 3. Metrics Collector (`metrics_collector.sh`)
A monitoring script that:
- Collects cluster metrics every minute
- Tracks pod counts, HPA status, and node counts
- Records node creation times for cost analysis
- Stores data in CSV format for analysis

## Prerequisites

- Kubernetes cluster
- kubectl configured with cluster access
- Python 3.6+
- Apache Bench (ab) installed
- Required Python packages:
  ```
  pip install schedule
  ```

## Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Configure your environment:
   - Update `HOST_URL` in `load_tester.py`
   - Adjust `MAX_REQUESTS` and `PEAK_HOUR` in `load_tester.py`
   - Modify `max_pods` in `proactive_scaler.py` if needed


## Usage

1. Start the metrics collector:
   ```bash
   chmod +x metrics_collector.sh
   ./metrics_collector.sh
   ```

2. Launch the load tester:
   ```bash
   python load_tester.py
   ```


## Data Files

The system generates several data files:
- `historical_data.csv`: Request patterns and error rates
- `metrics.csv`: Cluster metrics over time
- `nodes.csv`: Node creation and deletion tracking
- `load_test.log`: Load testing logs

## Monitoring

You can monitor the system's performance using:
```bash
# View current pod count
kubectl get pods -l app=simpleweb

# Check scaling history
tail -f metrics.csv

# Monitor load testing
tail -f load_test.log
```

## Customization

### Adjusting Load Patterns
In `load_tester.py`:
- `PEAK_HOUR`: Change peak traffic time
- `MAX_REQUESTS`: Modify maximum request count
- `SIGMA_MINUTES`: Adjust the spread of the traffic curve

### Modifying Scaling Behavior
In `proactive_scaler.py`:
- `LOOKBACK_DAYS`: Change historical data analysis period
- `max_pods`: Adjust maximum pod limit
- Modify `calculate_desired_pods()` for different scaling algorithms

## Troubleshooting

1. If load testing fails:
   - Verify `HOST_URL` is accessible
   - Check `load_test.log` for errors
   - Ensure Apache Bench is installed

2. If scaling isn't working:
   - Verify kubectl access
   - Check `historical_data.csv` exists
   - Ensure deployment name matches in `proactive_scaler.py`

## Contributing

Feel free to submit issues and enhancement requests!