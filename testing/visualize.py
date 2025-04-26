import os
import glob
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = "./data"
OUTPUT_DIR = "./plots"
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load all CSV files from data directory into a single DataFrame"""
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {DATA_DIR}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['datetime'].dt.strftime(DATE_FORMAT)
        df['time'] = df['datetime'].dt.strftime(TIME_FORMAT)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df):
    # removed_rows = df[df['latency_max_ms'] > 500]
    # removed_rows = df[df['target_rate'] < 8]
    # removed_rows += df[df['target_rate'] < 4]
    # for _, row in removed_rows.iterrows():
    #     if row['target_rate'] > 18:
    #         warnings.warn(
    #             f"Row with high latency_max ({row['latency_max_ms']} ms) "
    #             f"and target_rate ({row['target_rate']}) removed.")
    # df = df[~((df['target_rate'] < 20) & (df['latency_max_ms'] > 100))].copy()
    # df = df[].copy()

    # df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    # df['date'] = df['datetime'].dt.strftime(DATE_FORMAT)
    # df['time'] = df['datetime'].dt.strftime(TIME_FORMAT)

    return df


def plot_requests_over_time(df):
    """Plot requests per second over time with multiple days"""
    plt.figure(figsize=(15, 7))
    sns.lineplot(
        data=df,
        x='time',
        y='requests_per_sec',
        hue='date',
        marker='o',
        palette='tab10',
        errorbar=None
    )
    plt.title('Requests per Second Over Time')
    plt.xlabel('Time of Day')
    plt.ylabel('Requests/sec')
    plt.xticks(rotation=45)
    plt.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'requests_over_time.png'))
    plt.close()

def plot_requests_and_pods_over_time(df):
    """Plot requests and pods with dual y-axes"""
    plt.figure(figsize=(15, 8))
    
    # Create primary axis for requests
    ax1 = plt.gca()
    sns.lineplot(
        data=df,
        x='time',
        y='requests_per_sec',
        hue='date',
        marker='o',
        palette='tab10',
        errorbar=None,
        ax=ax1
    )
    
    # Create secondary axis for pods
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x='time',
        y='pods',
        hue='date',
        marker='s',
        palette='tab10',
        errorbar=None,
        ax=ax2,
        linestyle='--',
        legend=False
    )
    
    # Formatting
    ax1.set_title('Requests and Pods Over Time', pad=20)
    ax1.set_xlabel('Time of Day')
    ax1.set_ylabel('Requests/sec')
    ax2.set_ylabel('Pods Count')
    
    # Axis color matching
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, _ = ax2.get_legend_handles_labels()
    combined_labels = [f"{label} (Requests)" for label in labels1] + \
                     [f"{label} (Pods)" for label in labels1]
    ax1.legend(
        lines1 + lines2[:len(lines1)], 
        combined_labels,
        title='Date',
        bbox_to_anchor=(1.1, 1),
        loc='upper left'
    )
    
    plt.xticks(rotation=45)
    # lower the x-axis ticks to avoid overlap
    ax1.xaxis.set_major_locator(plt.MaxNLocator(24))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(24))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'requests_pods_over_time.png'))
    plt.close()

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_latency_distribution(df):
    """Plot latency percentiles over time with extreme outliers removed"""
    # Melt dataframe for better seaborn handling
    latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'p99.9_ms']
    # latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'latency_avg_ms']
    melted_df = df.melt(
        id_vars=['date', 'time'],
        value_vars=latency_metrics,
        var_name='percentile',
        value_name='latency'
    )

    # Plot
    plt.figure(figsize=(15, 7))
    sns.lineplot(
        data=melted_df,
        x='time',
        y='latency',
        hue='percentile',
        style='date',
        markers=True,
        dashes=False,
        palette='viridis',
        errorbar=None
    )
    plt.title('Latency Percentiles Over Time')
    plt.xlabel('Time of Day')
    plt.ylabel('Latency (ms)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_distribution.png'))
    plt.close()


def plot_latency_if_target_rate_high(df):
    """Plot latency percentiles over time only if target_rate > 10, using timestamp"""
    # latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'latency_avg_ms']
    latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'p99.9_ms']

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Filter based on target_rate
    filtered_df = df[df['target_rate'] > 4].copy()

    # Melt for seaborn
    melted_df = filtered_df.melt(
        id_vars=['date', 'datetime'],
        value_vars=latency_metrics,
        var_name='percentile',
        value_name='latency'
    )

    # Plot
    plt.figure(figsize=(15, 7))
    sns.lineplot(
        data=melted_df,
        x='datetime',
        y='latency',
        hue='percentile',
        style='date',
        markers=True,
        dashes=False,
        palette='viridis',
        errorbar=None
    )
    plt.title('Latency Percentiles (target_rate > 6)')
    plt.xlabel('Time')
    plt.ylabel('Latency (ms)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_target_rate_gt_6.png'))
    plt.close()

def plot_latency_by_3hr_chunks(df):
    """Create multiple latency plots split into 3-hour time chunks using timestamp"""
    # latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'latency_avg_ms']
    latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'p99.9_ms']
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour

    for start_hour in range(0, 24, 3):
        end_hour = start_hour + 3
        chunk_df = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)].copy()

        if chunk_df.empty:
            continue

        # Melt
        melted_df = chunk_df.melt(
            id_vars=['date', 'datetime'],
            value_vars=latency_metrics,
            var_name='percentile',
            value_name='latency'
        )


        # Plot
        plt.figure(figsize=(15, 7))
        sns.lineplot(
            data=melted_df,
            x='datetime',
            y='latency',
            hue='percentile',
            style='date',
            markers=True,
            dashes=False,
            palette='viridis',
            errorbar=None
        )
        plt.title(f'Latency Percentiles ({start_hour}:00–{end_hour}:00)')
        plt.xlabel('Time')
        plt.ylabel('Latency (ms)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        filename = f'latency_chunk_{start_hour:02d}_{end_hour:02d}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()


def plot_target_vs_actual(df):
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df,
        x='target_rate',
        y='requests_per_sec',
        hue='pods',
        size='total_requests',
        palette='viridis',
        alpha=0.7
    )
    
    # Add perfect correlation line
    max_rate = max(df['target_rate'].max(), df['requests_per_sec'].max())
    plt.plot([0, max_rate], [0, max_rate], 'r--', label='Ideal Performance')
    
    plt.title('Target Rate vs Actual Requests/sec\n(Bubble Size = Total Requests)')
    plt.xlabel('Target Rate (requests/sec)')
    plt.ylabel('Actual Requests/sec')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_vs_actual.png'))
    plt.close()

def plot_pods_requests_over_time(df):
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot Requests
    sns.lineplot(
        data=df,
        x='timestamp',
        y='requests_per_sec',
        color='#1f77b4',
        ax=ax1,
        label='Requests/sec'
    )
    ax1.set_ylabel('Requests/sec', color='#1f77b4')
    ax1.tick_params(axis='y', colors='#1f77b4')

    # Plot Pods on secondary axis
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x='timestamp',
        y='pods',
        color='#ff7f0e',
        ax=ax2,
        linestyle='--',
        label='Pods'
    )
    ax2.set_ylabel('Pods Count', color='#ff7f0e')
    ax2.tick_params(axis='y', colors='#ff7f0e')

    plt.title('Requests and Pod Scaling Over Time')
    ax1.set_xlabel('Time')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pods_requests_over_time.png'))
    plt.close()


def plot_latency_heatmap(df):
    plt.figure(figsize=(15, 8))
    latency_columns = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'p99.9_ms']
    
    # Create pivot table
    heatmap_data = df.pivot_table(
        index=pd.to_datetime(df['timestamp'], unit='s').dt.hour,
        columns=pd.to_datetime(df['timestamp'], unit='s').dt.minute,
        values=latency_columns
    )
    
    sns.heatmap(
        heatmap_data,
        cmap='viridis',
        cbar_kws={'label': 'Latency (ms)'}
    )
    
    plt.title('Latency Distribution Heatmap by Time of Day')
    plt.xlabel('Minute of Hour')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_heatmap.png'))
    plt.close()


def plot_requests_per_pod(df):
    df['requests_per_pod'] = df['requests_per_sec'] / df['pods']
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df,
        x='pods',
        y='requests_per_pod',
        hue='target_rate',
        palette='Spectral'
    )
    
    plt.title('Request Handling Efficiency per Pod')
    plt.xlabel('Number of Pods')
    plt.ylabel('Requests/sec per Pod')
    plt.legend(title='Target Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'requests_per_pod.png'))
    plt.close()


def plot_performance_quadrant(df):
    g = sns.JointGrid(
        data=df,
        x='requests_per_sec',
        y='latency_avg_ms',
        height=10
    )
    
    g.plot_joint(
        sns.scatterplot,
        hue=df['pods'],
        size=df['total_requests'],
        palette='plasma',
        alpha=0.7
    )
    
    g.plot_marginals(
        sns.histplot,
        kde=True,
        color='purple'
    )
    
    g.ax_joint.axhline(
        y=df['latency_avg_ms'].median(),
        color='r',
        linestyle='--',
        label='Median Latency'
    )
    
    g.ax_joint.axvline(
        x=df['requests_per_sec'].median(),
        color='g',
        linestyle='--',
        label='Median RPS'
    )
    
    plt.suptitle('Performance Quadrant Analysis\n(Color = Pods, Size = Total Requests)')
    g.ax_joint.set_xlabel('Requests/sec')
    g.ax_joint.set_ylabel('Average Latency (ms)')
    g.ax_joint.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'performance_quadrant.png'))
    plt.close()

def plot_latency_with_target_rate(df):
    """Plot latency percentiles over time with target_rate > 6 and overlay target rate"""
    # latency_metrics = ['p50_ms', 'p75_ms']
    # latency_metrics = ['p50_ms', 'p75_ms', 'latency_avg_ms']
    # latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'p99.9_ms']
    latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'p99.9_ms', 'latency_avg_ms']
    # latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'latency_avg_ms']
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    filtered_df = df[df['target_rate'] > 4].copy()

    melted_df = filtered_df.melt(
        id_vars=['date', 'datetime', 'target_rate'],
        value_vars=latency_metrics,
        var_name='percentile',
        value_name='latency'
    )

    plt.figure(figsize=(15, 7))
    ax = sns.lineplot(
        data=melted_df,
        x='datetime',
        y='latency',
        hue='percentile',
        style='date',
        markers=True,
        dashes=False,
        palette='viridis',
        errorbar=None
    )

    ax.set_title('Latency Percentiles with Target Rate Overlay (target_rate > 6)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Latency (ms)')
    ax.set_ylim(0, 500)                # ← limit primary y-axis to max 500
    plt.xticks(rotation=45)

    # Secondary axis for target_rate
    ax2 = ax.twinx()
    ax2.plot(filtered_df['datetime'], filtered_df['target_rate'], 'k--', label='Target Rate')
    ax2.set_ylabel('Target Rate (req/s)')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.05, 0.85), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_with_target_rate_overlay.png'))
    plt.close()


def plot_latency_by_3hr_chunks_with_target(df):
    """Create latency plots in 3-hour segments with target rate overlay"""
    latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'latency_avg_ms']

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour

    for start_hour in range(0, 24, 3):
        end_hour = start_hour + 3
        chunk_df = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)].copy()

        if chunk_df.empty:
            continue

        melted_df = chunk_df.melt(
            id_vars=['date', 'datetime', 'target_rate'],
            value_vars=latency_metrics,
            var_name='percentile',
            value_name='latency'
        )

        plt.figure(figsize=(15, 7))
        ax = sns.lineplot(
            data=melted_df,
            x='datetime',
            y='latency',
            hue='percentile',
            style='date',
            markers=True,
            dashes=False,
            palette='viridis',
            errorbar=None
        )

        ax.set_title(f'Latency Percentiles with Target Rate ({start_hour:02d}:00–{end_hour:02d})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Latency (ms)')
        ax.set_ylim(0, 500)                # ← limit primary y-axis to max 500
        plt.xticks(rotation=45)

        # Overlay target_rate
        ax2 = ax.twinx()
        ax2.plot(chunk_df['datetime'], chunk_df['target_rate'], 'k--', label='Target Rate')
        ax2.set_ylabel('Target Rate (req/s)')

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.legend(bbox_to_anchor=(1.05, 0.85), loc='upper left')
        plt.tight_layout()
        filename = f'latency_chunk_with_target_{start_hour:02d}_{end_hour:02d}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()

def plot_latency_by_1hr_chunks_with_target_offset(df):
    """PREDICTIVE: latency plots in 1-hour windows (e.g., 0:30–1:30) using existing hour and minute columns."""
    latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'latency_avg_ms']
    # latency_metrics = ['p50_ms', 'p75_ms']
    # latency_metrics = ['p50_ms', 'p75_ms', 'latency_avg_ms']
    # latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'p99.9_ms']
    # latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'p99.9_ms', 'latency_avg_ms']
    # latency_metrics = ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms', 'latency_avg_ms']

    # Assume df already has 'hour' and 'minute'
    for start_hour in range(0, 24, 1):
        start_minute = 30  # You want to start at :30
        end_hour = (start_hour + 1) % 24
        end_minute = 30

        # Select rows between start and end
        chunk_df = df[
            ((df['hour'] == start_hour) & (df['minute'] >= start_minute)) |
            ((df['hour'] == end_hour) & (df['minute'] < end_minute))
        ].copy()

        if chunk_df.empty:
            continue

        melted_df = chunk_df.melt(
            id_vars=['date', 'datetime', 'target_rate'],
            value_vars=latency_metrics,
            var_name='percentile',
            value_name='latency'
        )

        plt.figure(figsize=(15, 7))
        ax = sns.lineplot(
            data=melted_df,
            x='datetime',
            y='latency',
            hue='percentile',
            style='date',
            markers=True,
            dashes=False,
            palette='viridis',
            errorbar=None
        )

        ax.set_title(f'Latency Percentiles with Target Rate ({start_hour:02d}:00–{end_hour:02d})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Latency (ms)')
        ax.set_ylim(0, 500)                # ← limit primary y-axis to max 500
        plt.xticks(rotation=45)

        # Overlay target_rate
        ax2 = ax.twinx()
        ax2.plot(chunk_df['datetime'], chunk_df['target_rate'], 'k--', label='Target Rate')
        ax2.set_ylabel('Target Rate (req/s)')

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.legend(bbox_to_anchor=(1.05, 0.85), loc='upper left')
        plt.tight_layout()
        filename = f'latency_chunk_with_target_{start_hour:02d}_{end_hour:02d}_offset.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()


def generate_report():
    """Main function to generate all visualizations"""
    sns.set_theme(style="whitegrid", palette="pastel")
    
    # Load data
    df = load_data()
    df = preprocess_data(df)

    plot_latency_if_target_rate_high(df)
    plot_latency_by_1hr_chunks_with_target_offset(df)
    # plot_latency_by_3hr_chunks(df)
    plot_latency_with_target_rate(df)
    plot_latency_by_3hr_chunks_with_target(df)


    
    # Generate plots
    plot_requests_over_time(df)
    plot_latency_distribution(df)
    plot_requests_and_pods_over_time(df)

    plot_target_vs_actual(df)
    plot_pods_requests_over_time(df)
    plot_latency_heatmap(df)
    plot_requests_per_pod(df)
    plot_performance_quadrant(df)

    print(f"Visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_report()