import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import math
import random
import matplotlib.dates as mdates

# Configuration
PEAK_HOURS = [2, 6, 7, 10, 14, 16, 18, 21]  # Multiple peaks at 8AM, 12PM, 6PM
PEAK_WEIGHTS = [random.uniform(0.9, 1.0) for _ in range(len(PEAK_HOURS))]
SIGMA_MINUTES = 55  # Peak width
MAX_RATE = 45       # Maximum requests/sec
MIN_RATE = 1        # Minimum requests/sec
OUTPUT_CSV = "daily_traffic.csv"
PLOT_FILE = "daily_traffic.png"
OUTPUT_DIR = "./plots"
# Ensure output directory exists
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# Set the output file path
PLOT_FILE = os.path.join(OUTPUT_DIR, PLOT_FILE)
# Set the output CSV file path
OUTPUT_CSV = os.path.join(OUTPUT_DIR, OUTPUT_CSV)

def calculate_traffic_rate(current_time):
    """Multi-peak traffic calculation"""
    current_minutes = current_time.hour * 60 + current_time.minute
    combined_rate = 0.0
    
    for peak_hour, weight in zip(PEAK_HOURS, PEAK_WEIGHTS):
        peak_minutes = peak_hour * 60
        delta = current_minutes - peak_minutes
        exponent = -(delta ** 2) / (2 * (SIGMA_MINUTES ** 2))
        peak_contribution = weight * math.exp(exponent)
        # Normalize to MAX_RATE * weight at the peak
        combined_rate += peak_contribution * MAX_RATE

    # Add noise
    
    noise = 0
    noise = random.uniform(-0.1, 0.1)
    rate = int(combined_rate * (1 + noise))

    return max(rate, MIN_RATE)


def generate_daily_data():
    """Create dataframe with traffic rates for every minute of a day"""
    start_time = datetime.strptime("00:00", "%H:%M")
    times = [start_time + timedelta(minutes=i) for i in range(1440)]
    
    data = []
    for t in times:
        rate = calculate_traffic_rate(t)
        data.append({
            'time': t.strftime("%H:%M"),
            'hour': t.hour,
            'minute': t.minute,
            'rate': rate
        })
    
    return pd.DataFrame(data)

def visualize_daily_traffic(df):
    """Create visualization of daily traffic pattern"""
    plt.figure(figsize=(15, 7))
    sns.set_style("whitegrid")
    
    df['parsed_time'] = pd.to_datetime(df['time'], format='%H:%M')

    # Now plot — keep only_time as datetime64
    ax = sns.lineplot(
        x='parsed_time',
        y='rate',
        data=df,
        color='#2c7fb8',
        linewidth=2
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    
    # Formatting
    ax.set_title("Daily Traffic Pattern Simulation", fontsize=16, pad=20)
    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Requests per Second", fontsize=12)
    ax.set_ylim(0, MAX_RATE * 1.3)
    
    # X-axis formatting
    ax.xaxis.set_major_locator(plt.MaxNLocator(24))
    plt.xticks(rotation=45, ha='right')
    
    # Add peak markers
    for peak in PEAK_HOURS:
        plt.axvline(
            x=pd.to_datetime(f"{peak:02d}:00", format='%H:%M'),
            color='red',
            linestyle='--',
            alpha=0.3
        )
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Generate and save data
    df = generate_daily_data()
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Create visualization
    visualize_daily_traffic(df)
    
    print(f"Data saved to {OUTPUT_CSV}")
    print(f"Visualization saved to {PLOT_FILE}")