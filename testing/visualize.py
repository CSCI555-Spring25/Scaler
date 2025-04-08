import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import datetime

# Configuration
HISTORICAL_DATA_FILE = "historical_data.csv"
POD_COUNTS_FILE = "pod_counts.json"
METRICS_FILE = "metrics.csv"
OUTPUT_DIR = "graphs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_historical_data():
    """Load and process historical traffic data"""
    if not os.path.exists(HISTORICAL_DATA_FILE):
        return None
    
    df = pd.read_csv(HISTORICAL_DATA_FILE)
    if df.empty:
        return None
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def load_pod_counts():
    """Load pod count data"""
    if not os.path.exists(POD_COUNTS_FILE):
        return None
    
    with open(POD_COUNTS_FILE, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    if not data.get('data'):
        return None
        
    entries = data['data']
    df = pd.DataFrame(entries)
    
    # Convert timestamp to datetime
    today = datetime.datetime.now().date()
    df['datetime'] = df['timestamp'].apply(
        lambda x: datetime.datetime.combine(
            today, 
            datetime.time(
                hour=int(x.split(':')[0]), 
                minute=int(x.split(':')[1])
            )
        )
    )
    
    return df

def load_metrics():
    """Load system metrics data"""
    if not os.path.exists(METRICS_FILE):
        return None
    
    df = pd.read_csv(METRICS_FILE)
    if df.empty:
        return None
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def plot_traffic_over_time(df):
    """Plot traffic requests over time"""
    if df is None or df.empty:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['intended_requests'], 'b-', label='Intended Requests')
    
    if 'complete_requests' in df.columns:
        plt.plot(df['datetime'], df['complete_requests'].astype(int), 'g-', label='Completed Requests')
    
    if 'failed_requests' in df.columns and df['failed_requests'].astype(int).sum() > 0:
        plt.plot(df['datetime'], df['failed_requests'].astype(int), 'r-', label='Failed Requests')
    
    plt.title('Traffic Load Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Requests')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/traffic_over_time.png')
    plt.close()

def plot_pod_counts(pod_df, metrics_df=None):
    """Plot pod counts over time"""
    plt.figure(figsize=(12, 6))
    
    # Plot pod counts from JSON data
    if pod_df is not None and not pod_df.empty:
        plt.plot(pod_df['datetime'], pod_df['podCount'], 'b-', marker='o', label='Pod Count (JSON)')
    
    # Also plot pod counts from metrics if available
    if metrics_df is not None and not metrics_df.empty:
        plt.plot(metrics_df['datetime'], metrics_df['pod_count'], 'g--', marker='x', label='Pod Count (Metrics)')
        
        if 'hpa_desired' in metrics_df.columns:
            plt.plot(metrics_df['datetime'], metrics_df['hpa_desired'], 'r-.', label='HPA Desired')
    
    plt.title('Pod Count Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Pods')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pod_counts.png')
    plt.close()

def plot_performance_metrics(df):
    """Plot performance metrics over time"""
    if df is None or df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Requests per second
    if 'requests_per_second' in df.columns:
        axes[0, 0].plot(df['datetime'], df['requests_per_second'].astype(float), 'b-')
        axes[0, 0].set_title('Requests per Second')
        axes[0, 0].set_ylabel('Requests/s')
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Time per request
    if 'time_per_request_mean' in df.columns:
        axes[0, 1].plot(df['datetime'], df['time_per_request_mean'].astype(float), 'g-')
        axes[0, 1].set_title('Time per Request (Mean)')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Latency percentiles
    percentiles = ['percentage_50', 'percentage_90', 'percentage_99']
    colors = ['b-', 'g-', 'r-']
    labels = ['50th', '90th', '99th']
    
    for pctl, color, label in zip(percentiles, colors, labels):
        if pctl in df.columns:
            axes[1, 0].plot(df['datetime'], df['total_mean'].astype(float), color, label=label)
    
    axes[1, 0].set_title('Response Time Percentiles')
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Traffic vs Pods
    if 'intended_requests' in df.columns:
        ax1 = axes[1, 1]
        color = 'b'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Requests', color=color)
        ax1.plot(df['datetime'], df['intended_requests'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=45)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Pod count on secondary y-axis
        pod_df = load_pod_counts()
        if pod_df is not None and not pod_df.empty:
            ax2 = ax1.twinx()
            color = 'r'
            ax2.set_ylabel('Pods', color=color)
            ax2.plot(pod_df['datetime'], pod_df['podCount'], color=color, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)
        
        axes[1, 1].set_title('Traffic vs Pods')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/performance_metrics.png')
    plt.close()

def plot_correlation(df, pod_df):
    """Plot correlation between traffic and pod count"""
    if df is None or df.empty or pod_df is None or pod_df.empty:
        return
    
    # Merge dataframes on closest datetime
    df_merged = pd.merge_asof(
        df.sort_values('datetime'),
        pod_df.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )
    
    if df_merged.empty or 'podCount' not in df_merged.columns:
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_merged['intended_requests'], df_merged['podCount'], alpha=0.6)
    
    # Add trend line
    if len(df_merged) > 1:
        z = np.polyfit(df_merged['intended_requests'], df_merged['podCount'], 1)
        p = np.poly1d(z)
        plt.plot(df_merged['intended_requests'], p(df_merged['intended_requests']), 'r--', alpha=0.8)
    
    plt.title('Correlation: Traffic vs Pod Count')
    plt.xlabel('Request Count')
    plt.ylabel('Number of Pods')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/traffic_pods_correlation.png')
    plt.close()

def generate_daily_pattern():
    """Generate visualization of the daily pattern with all peak hours"""
    hours = np.arange(0, 24, 0.25)  # 15-minute intervals
    
    # Generate traffic pattern for visualization
    def traffic_pattern(hour, peak_hours=[8, 12, 18], peak_weights=[0.6, 0.8, 1.0], sigma=1):
        total = 0
        for peak, weight in zip(peak_hours, peak_weights):
            delta = min(abs(hour - peak), 24 - abs(hour - peak))
            total += weight * np.exp(-(delta ** 2) / (2 * sigma ** 2))
        return total
    
    traffic = [traffic_pattern(h) for h in hours]
    
    plt.figure(figsize=(12, 6))
    plt.plot(hours, traffic, 'b-')
    
    # Mark peak hours
    peak_hours = [8, 12, 18]
    peak_names = ['Morning', 'Midday', 'Evening']
    for peak, name in zip(peak_hours, peak_names):
        plt.axvline(x=peak, color='r', linestyle='--', alpha=0.6)
        plt.text(peak, 0.05, name, rotation=90, verticalalignment='bottom')
    
    plt.title('Daily Traffic Pattern Template')
    plt.xlabel('Hour of Day')
    plt.ylabel('Relative Traffic Intensity')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/daily_pattern_template.png')
    plt.close()

def create_dashboard():
    """Create a simple HTML dashboard with all graphs"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kubernetes Scaling Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .graph-container { margin-bottom: 30px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Kubernetes Scaling Dashboard</h1>
        
        <div class="graph-container">
            <h2>Traffic Pattern</h2>
            <img src="daily_pattern_template.png" alt="Daily Pattern Template">
        </div>
        
        <div class="graph-container">
            <h2>Traffic Over Time</h2>
            <img src="traffic_over_time.png" alt="Traffic Over Time">
        </div>
        
        <div class="graph-container">
            <h2>Pod Counts</h2>
            <img src="pod_counts.png" alt="Pod Counts">
        </div>
        
        <div class="graph-container">
            <h2>Performance Metrics</h2>
            <img src="performance_metrics.png" alt="Performance Metrics">
        </div>
        
        <div class="graph-container">
            <h2>Traffic vs Pods Correlation</h2>
            <img src="traffic_pods_correlation.png" alt="Traffic vs Pods Correlation">
        </div>
        
        <p>Last updated: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </body>
    </html>
    """
    
    with open(f'{OUTPUT_DIR}/dashboard.html', 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    print("Generating visualizations...")
    
    # Load data
    historical_df = load_historical_data()
    pod_df = load_pod_counts()
    metrics_df = load_metrics()
    
    # Generate template pattern
    generate_daily_pattern()
    
    # Create visualizations
    plot_traffic_over_time(historical_df)
    plot_pod_counts(pod_df, metrics_df)
    plot_performance_metrics(historical_df)
    plot_correlation(historical_df, pod_df)
    
    # Create dashboard
    create_dashboard()
    
    print(f"Visualizations completed. Open {OUTPUT_DIR}/dashboard.html to view the dashboard.")