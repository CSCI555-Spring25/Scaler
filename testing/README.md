# Testing Framework for Predictive Autoscaler

This directory contains comprehensive testing tools and utilities for the Predictive Autoscaler project. The testing framework includes load generation, data analysis, visualization, and monitoring capabilities.

## Directory Structure

```
testing/
├── README.md                    # This file
├── config/                      # Configuration files
│   └── requirements.txt         # Python dependencies
├── docs/                        # Documentation
│   ├── instructions.md          # Quick start instructions
│   └── installation.txt         # Installation guide
├── scripts/                     # All executable scripts
│   ├── load_testing/           # Load generation scripts
│   │   ├── load_tester.py      # Main load testing tool
│   │   ├── load_tester_utils.py # Load testing utilities
│   │   ├── fast_load.py        # Fast load generation
│   │   ├── daily_traffic.py    # Daily traffic simulation
│   │   └── fast_load_test.py   # Quick load tests
│   ├── analysis/               # Data analysis and visualization
│   │   ├── visualize.py        # Data visualization
│   │   ├── visualize_final.py  # Final analysis visualization
│   │   └── log_analyzer.py     # Log analysis tools
│   └── utilities/              # Helper scripts and tools
│       ├── csv_to_json.py      # Data conversion utility
│       ├── test_with_traffic_1_interval.sh # Automated test script
│       └── metrics_collector/  # Metrics collection tools
├── sample_data/                # Sample and test data
│   ├── json/                   # JSON format test data
│   │   ├── traffic_1_interval.json
│   │   └── traffic_5_interval.json
│   └── csv/                    # CSV format test data
│       └── *.csv files
├── results/                    # Test results and outputs
│   ├── plots/                  # Generated visualizations
│   ├── reports/                # Analysis reports
│   └── logs/                   # Execution logs
└── archive/                    # Historical data and backups
```

## Quick Start

### Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl configured with cluster access
- Python 3.7+
- Apache Bench (ab) installed

### Setup

1. Install Python dependencies:
   ```bash
   pip install -r config/requirements.txt
   ```

2. Configure your environment:
   - Update HOST_URL in load testing scripts
   - Adjust scaling parameters as needed

### Running Tests

#### Automated Testing
Use the provided automation script for quick testing:
```bash
cd scripts/utilities
./test_with_traffic_1_interval.sh
```

#### Manual Testing

1. **Start Load Testing:**
   ```bash
   cd scripts/load_testing
   python load_tester.py
   ```

2. **Monitor Metrics:**
   ```bash
   cd scripts/utilities/metrics_collector
   ./metrics_collector.sh
   ```

3. **Analyze Results:**
   ```bash
   cd scripts/analysis
   python visualize.py
   ```

## Components

### Load Testing (`scripts/load_testing/`)

- **`load_tester.py`**: Main load generation tool with realistic traffic patterns
- **`load_tester_utils.py`**: Utility functions for load testing
- **`fast_load.py`**: Quick load generation for testing
- **`daily_traffic.py`**: Simulates daily traffic patterns
- **`fast_load_test.py`**: Rapid load testing scenarios

### Analysis Tools (`scripts/analysis/`)

- **`visualize.py`**: Creates charts and graphs from test data
- **`visualize_final.py`**: Comprehensive analysis and final reports
- **`log_analyzer.py`**: Analyzes log files and extracts metrics

### Utilities (`scripts/utilities/`)

- **`csv_to_json.py`**: Converts CSV data to JSON format
- **`test_with_traffic_1_interval.sh`**: Automated test execution
- **`metrics_collector/`**: Tools for collecting Kubernetes metrics

## Data Files

### Sample Data (`sample_data/`)
- **JSON files**: Pre-generated traffic patterns for testing
- **CSV files**: Historical load test data

### Results (`results/`)
- **Plots**: Generated visualizations and charts
- **Reports**: Analysis reports in text format
- **Logs**: Execution logs and debug information

## Monitoring and Analysis

### Real-time Monitoring
```bash
# View controller logs
kubectl logs -f deployment/predictive-autoscaler-controller

# Monitor HPA status
kubectl get hpa -w

# Check pod scaling
kubectl get pods -l app=simpleweb -w
```

### Post-test Analysis
```bash
# Generate visualizations
cd scripts/analysis
python visualize_final.py

# Analyze scaling patterns
python log_analyzer.py
```

## Customization

### Load Patterns
Modify traffic patterns in `scripts/load_testing/`:
- Adjust `PEAK_HOUR` for different peak times
- Change `MAX_REQUESTS` for load intensity
- Modify `SIGMA_MINUTES` for traffic distribution

### Analysis Parameters
Configure analysis in `scripts/analysis/`:
- Update time ranges for analysis
- Adjust visualization parameters
- Modify report generation settings

## Best Practices

1. **Before Testing:**
   - Ensure cluster has sufficient resources
   - Verify predictive autoscaler is running
   - Check baseline metrics

2. **During Testing:**
   - Monitor cluster resource usage
   - Watch for any error conditions
   - Document test parameters

3. **After Testing:**
   - Analyze generated reports
   - Archive important results
   - Clean up test resources

## Troubleshooting

### Common Issues

1. **Load testing fails:**
   - Verify target service is accessible
   - Check network connectivity
   - Ensure Apache Bench is installed

2. **Metrics collection stops:**
   - Verify kubectl access
   - Check cluster connectivity
   - Ensure sufficient disk space

3. **Analysis scripts error:**
   - Check data file formats
   - Verify Python dependencies
   - Ensure sufficient memory

### Getting Help

- Check logs in `results/logs/`
- Review error messages in output files
- Verify configuration in `config/`

## Contributing

When adding new testing tools:
1. Place scripts in appropriate subdirectories
2. Update this README with new components
3. Add configuration files to `config/`
4. Document usage and dependencies 