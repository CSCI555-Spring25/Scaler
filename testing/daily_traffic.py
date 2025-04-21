import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import math, random
import os

fall_sigma_min = 12
plateau_min = 12
# Configuration

PEAK_PARAMS = [
    (i * 60, 1.0, i, fall_sigma_min - (i // 2.5), plateau_min - (i // 2.5))
    for i in range(1, 22)
]

# generates:
# PEAK_PARAMS = [
#     # (hour, weight, rise_sigma_min, fall_sigma_min, plateau_min)
#     (1*60, 1.0,  1,  fall_sigma_min,  plateau_min ),   # e.g. 7AM peak, fall_sigma_min‑min rise, 20‑min flat, fall_sigma_min‑min fall
#     (2*60, 1.0,  2, fall_sigma_min,  plateau_min ),   # noon
#     (3*60, 1.0,  3,  fall_sigma_min,  plateau_min ),   # 6PM
#     ...
# ]
MAX_RATE = 125       # Maximum requests/sec
MIN_RATE = 1        # Minimum requests/sec


OUTPUT_CSV = "daily_traffic.csv"
PLOT_FILE = "daily_traffic.png"
OUTPUT_DIR = "./plots"
# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# Set the output file path
PLOT_FILE = os.path.join(OUTPUT_DIR, PLOT_FILE)
# Set the output CSV file path
OUTPUT_CSV = os.path.join(OUTPUT_DIR, OUTPUT_CSV)


_noise_prev = 0.0
def correlated_noise(alpha=0.8, scale=0.02):
    global _noise_prev
    e = random.gauss(0, scale)
    _noise_prev = alpha*_noise_prev + e
    return _noise_prev

def baseline_multiplier(dt):
    # weekends 40% lower
    # return 0.6 if dt.weekday() >= 5 else 1.0
    return 1.0

def calculate_traffic_rate(dt: datetime):
    t_min = dt.hour*60 + dt.minute
    total_weight = 0.0

    for peak_min, weight, rise_s, fall_s, plateau in PEAK_PARAMS:
        total_weight += peak_contribution(
            t_min, peak_min, weight,
            rise_s, fall_s, plateau
        )

    # Scale to rate, apply baseline and noise
    noise = correlated_noise()
    rate = total_weight * MAX_RATE * baseline_multiplier(dt) * (1 + noise)
    return max(int(rate), MIN_RATE)

def peak_contribution(t_min, peak_min, weight,
                      rise_sigma, fall_sigma, plateau):
    dt = t_min - peak_min
    # plateau region
    if abs(dt) <= plateau/2:
        return weight
    # rising edge
    if -plateau/2 - 3*rise_sigma < dt < -plateau/2:
        x = dt + plateau/2
        return weight * math.exp(-x*x/(2*rise_sigma**2))
    # falling edge
    if plateau/2 < dt < plateau/2 + 3*fall_sigma:
        x = dt - plateau/2
        return weight * math.exp(-x*x/(2*fall_sigma**2))
    return 0.0


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
    ax.set_title("Daily Traffic Pattern Simulation", fontsize=16, pad=fall_sigma_min)
    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Requests per Second", fontsize=12)
    ax.set_ylim(0, MAX_RATE * 1.3)
    
    # X-axis formatting
    ax.xaxis.set_major_locator(plt.MaxNLocator(24))
    plt.xticks(rotation=45, ha='right')

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