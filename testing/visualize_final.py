import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
from itertools import combinations

# Configuration
DATA_DIR = "./final_data"
OUTPUT_DIR = "./analysis_plots"
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load predictive and reactive datasets without datetime processing"""
    predictive_path = os.path.join(DATA_DIR, "predictive.csv")
    reactive_path = os.path.join(DATA_DIR, "reactive.csv")
    
    predictive_df = pd.read_csv(predictive_path)
    reactive_df = pd.read_csv(reactive_path)
    
    # Create synthetic time identifier
    for df in [predictive_df, reactive_df]:
        df['time_id'] = df['hour'] + df['minute']/60
        df['time_window'] = df['hour'].astype(str) + ':' + df['minute'].astype(str).str.zfill(2)
    
    return predictive_df, reactive_df

# Helper function for time window creation
def create_offset_windows(start_hour=4, start_minute=30, window_size=60):
    """Generate 30-minute offset 1-hour windows"""
    windows = []
    current_hour = start_hour
    current_minute = start_minute
    
    while (current_hour < 23) or (current_hour == 23 and current_minute <= 30):
        end_hour = current_hour
        end_minute = current_minute + window_size
        if end_minute >= 60:
            end_hour += 1
            end_minute %= 60
        
        windows.append((
            f"{current_hour:02d}:{current_minute:02d}",
            f"{end_hour:02d}:{end_minute:02d}"
        ))
        
        # Move window by 30 minutes
        current_hour += 0 if current_minute + 30 < 60 else 1
        current_minute = (current_minute + 30) % 60
    
    return windows

def plot_individual_latency(ax, chunk, title, start_time, end_time, latency_metrics):
    """Helper function for individual latency plotting with dynamic metrics"""
    if chunk.empty:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        return
    
    # Create color palette for latency metrics
    colors = sns.color_palette("husl", n_colors=len(latency_metrics))
    
    # Plot each latency metric
    for idx, metric in enumerate(latency_metrics):
        sns.lineplot(data=chunk, x='time_id', y=metric, ax=ax,
                    label=metric.replace('_ms', ''), color=colors[idx])
    
    # Create twin axis for target rate
    ax2 = ax.twinx()
    ax2.plot(chunk['time_id'], chunk['target_rate'], 
            color='black', linestyle='--', linewidth=1.5, label='Target Rate')
    
    # Axis labeling
    ax.set_title(f"{title} System: {start_time}-{end_time}")
    ax.set_xlabel("Time (Decimal Hours)")
    ax.set_ylabel("Latency (ms)", color=colors[0])
    ax2.set_ylabel("Target Rate (req/s)", color='black')
    
    # Style axes
    ax.tick_params(axis='y', colors=colors[0])
    ax2.tick_params(axis='y', colors='black')
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

def plot_combined_latency(ax, p_chunk, r_chunk, start, end, latency_metrics):
    """Helper function for combined latency plotting with dynamic metrics"""
    # Create twin axes
    ax2 = ax.twinx()
    colors = sns.color_palette("husl", n_colors=len(latency_metrics))
    
    # Plot latency metrics
    for idx, metric in enumerate(latency_metrics):
        # Predictive system
        if not p_chunk.empty:
            sns.lineplot(data=p_chunk, x='time_id', y=metric, ax=ax,
                        color=colors[idx], linestyle='-',
                        label=f'Predictive {metric.replace("_ms", "")}')
        
        # Reactive system
        if not r_chunk.empty:
            sns.lineplot(data=r_chunk, x='time_id', y=metric, ax=ax,
                        color=colors[idx], linestyle='--',
                        label=f'Reactive {metric.replace("_ms", "")}')
    
    # Plot target rates on secondary axis
    if not p_chunk.empty:
        ax2.plot(p_chunk['time_id'], p_chunk['target_rate'],
                color='navy', linestyle=':', label='Predictive Target')
    
    if not r_chunk.empty:
        ax2.plot(r_chunk['time_id'], r_chunk['target_rate'],
                color='darkorange', linestyle=':', label='Reactive Target')
    
    # Styling and labeling
    ax.set_title(f"Combined Analysis: {start}-{end}")
    ax.set_xlabel("Time (Decimal Hours)")
    ax.set_ylabel("Latency (ms)", color=colors[0])
    ax2.set_ylabel("Target Rate (req/s)", color='navy')
    
    # Color ticks appropriately
    ax.tick_params(axis='y', colors=colors[0])
    ax2.tick_params(axis='y', colors='navy')
    
    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

def plot_latency_by_1hr_chunks_with_target_offset(predictive_df, reactive_df):
    """Create multiple plot variations with different metric combinations"""
    metric_groups = [
        ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms'],
        ['p50_ms', 'p75_ms', 'p90_ms', 'latency_avg_ms'],
        ['p50_ms', 'p75_ms'],
        ['p50_ms', 'latency_avg_ms']
    ]
    
    windows = create_offset_windows()
    
    # Create main directory for this analysis
    analysis_dir = os.path.join(OUTPUT_DIR, "latency_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    for group_idx, latency_metrics in enumerate(metric_groups):
        # Create subdirectory for this metric group
        group_name = "_".join([m.replace("_ms", "") for m in latency_metrics])
        group_dir = os.path.join(analysis_dir, f"metrics_{group_name}")
        os.makedirs(group_dir, exist_ok=True)
        
        for window in windows:
            start_time, end_time = window
            start_hour, start_minute = map(int, start_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))
            
            # Filter data for both datasets
            p_mask = ((predictive_df['hour'] == start_hour) & (predictive_df['minute'] >= start_minute)) | \
                     ((predictive_df['hour'] == end_hour) & (predictive_df['minute'] < end_minute))
            
            r_mask = ((reactive_df['hour'] == start_hour) & (reactive_df['minute'] >= start_minute)) | \
                     ((reactive_df['hour'] == end_hour) & (reactive_df['minute'] < end_minute))
            
            p_chunk = predictive_df[p_mask].copy()
            r_chunk = reactive_df[r_mask].copy()
            
            if p_chunk.empty and r_chunk.empty:
                continue
            
            # Create figures with metric group in filename
            plt.figure(figsize=(20, 12))
            gs = GridSpec(2, 2, figure=plt.gcf())
            
            # Combined plot with current metric group
            ax_combined = plt.subplot(gs[:, 0])
            plot_combined_latency(ax_combined, p_chunk, r_chunk, 
                                start_time, end_time, latency_metrics)
            
            # Individual plots with full metrics
            ax_predictive = plt.subplot(gs[0, 1])
            plot_individual_latency(ax_predictive, p_chunk, "Predictive",
                                   start_time, end_time, latency_metrics)
            
            ax_reactive = plt.subplot(gs[1, 1])
            plot_individual_latency(ax_reactive, r_chunk, "Reactive",
                                   start_time, end_time, latency_metrics)
            
            # Save with time window identifier in metric group subdirectory
            filename = f"latency_{start_time.replace(':','')}-{end_time.replace(':','')}.png"
            plt.savefig(os.path.join(group_dir, filename))
            plt.close()
            
            # Also save a CSV of the data used for this plot
            # combined_chunk = pd.concat([
            #     p_chunk.assign(system='predictive'),
            #     r_chunk.assign(system='reactive')
            # ])
            # csv_filename = f"data_{start_time.replace(':','')}-{end_time.replace(':','')}.csv"
            # combined_chunk.to_csv(os.path.join(group_dir, csv_filename), index=False)

def plot_pods_with_target_rate(predictive_df, reactive_df):
    """Plot pod counts with target rate comparison"""
    plt.figure(figsize=(18, 10))
    
    # Combined plot
    plt.subplot(2, 1, 1)
    sns.lineplot(data=predictive_df, x='time_id', y='pods', 
                label='Predictive Pods', color='blue')
    sns.lineplot(data=reactive_df, x='time_id', y='pods', 
                label='Reactive Pods', color='orange')
    plt.plot(predictive_df['time_id'], predictive_df['target_rate'],
            label='Predictive Target', color='navy', linestyle='--')
    plt.plot(reactive_df['time_id'], reactive_df['target_rate'],
            label='Reactive Target', color='darkorange', linestyle='--')
    plt.title("Pod Counts and Target Rates Comparison")
    plt.legend()
    
    # Individual pod plots
    plt.subplot(2, 2, 3)
    sns.lineplot(data=predictive_df, x='time_id', y='pods', color='blue')
    plt.title("Predictive Pod Scaling")
    
    plt.subplot(2, 2, 4)
    sns.lineplot(data=reactive_df, x='time_id', y='pods', color='orange')
    plt.title("Reactive Pod Scaling")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pods_target_comparison.png"))
    plt.close()

def plot_requests_over_time(predictive_df, reactive_df):
    """Visualize request patterns with multiple perspectives"""
    # Overlay plot
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=predictive_df, x='time_id', y='requests_per_sec',
                label='Predictive', color='blue')
    sns.lineplot(data=reactive_df, x='time_id', y='requests_per_sec',
                label='Reactive', color='orange')
    plt.title("Request Rate Comparison")
    plt.savefig(os.path.join(OUTPUT_DIR, "requests_overlay.png"))
    plt.close()
    
    # Faceted plot
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    sns.lineplot(data=predictive_df, x='time_id', y='requests_per_sec', color='blue')
    plt.title("Predictive Request Pattern")
    
    plt.subplot(2, 1, 2)
    sns.lineplot(data=reactive_df, x='time_id', y='requests_per_sec', color='orange')
    plt.title("Reactive Request Pattern")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "requests_faceted.png"))
    plt.close()

def plot_target_vs_actual(predictive_df, reactive_df):
    """Comparative target vs actual performance"""
    plt.figure(figsize=(14, 8))
    
    # Predictive
    hexbin = plt.hexbin(predictive_df['target_rate'], predictive_df['requests_per_sec'],
                       gridsize=30, cmap='Blues', alpha=0.7, 
                       label='Predictive Density')
    plt.colorbar(hexbin, label='Data Density (Predictive)')
    
    # Reactive
    hexbin = plt.hexbin(reactive_df['target_rate'], reactive_df['requests_per_sec'],
                       gridsize=30, cmap='Oranges', alpha=0.7,
                       label='Reactive Density')
    plt.colorbar(hexbin, label='Data Density (Reactive)')
    
    max_rate = max(predictive_df['target_rate'].max(), reactive_df['target_rate'].max())
    plt.plot([0, max_rate], [0, max_rate], 'k--', label='Ideal Performance')
    
    plt.title("Target vs Actual Request Rate Comparison")
    plt.xlabel("Target Rate (requests/sec)")
    plt.ylabel("Actual Requests/sec")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "target_vs_actual.png"))
    plt.close()

def plot_requests_per_pod(predictive_df, reactive_df):
    """Comparative request efficiency analysis"""
    # Bin target rates
    bins = [0, 4, 8, 12, 16, 20]
    labels = ['0-4', '5-8', '9-12', '13-16', '17-20']
    
    for df, name in zip([predictive_df, reactive_df], ['predictive', 'reactive']):
        df = df.copy()
        df['target_bin'] = pd.cut(df['target_rate'], bins=bins, labels=labels)
        df['requests_per_pod'] = df['requests_per_sec'] / df['pods']
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='pods', y='requests_per_pod', hue='target_bin',
                   palette='viridis')
        
        # Add statistical annotations
        pod_groups = df.groupby('pods')['requests_per_pod']
        p_values = []
        for (pod1, group1), (pod2, group2) in combinations(pod_groups, 2):
            _, p = stats.ttest_ind(group1, group2, nan_policy='omit')
            p_values.append((pod1, pod2, p))
        
        # Annotate significant pairs
        y_max = df['requests_per_pod'].max() * 1.1
        for i, (pod1, pod2, p) in enumerate(p_values):
            if p < 0.05:
                plt.plot([pod1, pod2], [y_max - i*0.5, y_max - i*0.5], 'k-')
                plt.text((pod1+pod2)/2, y_max - i*0.5 + 0.1,
                        f'p={p:.2e}', ha='center')
        
        plt.title(f"Request Handling Efficiency - {name.title()}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"requests_per_pod_{name}.png"))
        plt.close()

def plot_performance_quadrant(predictive_df, reactive_df):
    """Quadrant analysis with dynamic median calculation"""
    def create_quadrant_plot(df, title):
        # Calculate dynamic medians
        rps_median = df['requests_per_sec'].median()
        latency_median = df['latency_avg_ms'].median()
        
        g = sns.JointGrid(data=df, x='requests_per_sec', y='latency_avg_ms', height=10)
        g.plot_joint(sns.kdeplot, fill=True, cmap='viridis', alpha=0.6)
        g.plot_joint(sns.scatterplot, hue=df['pods'], palette='coolwarm', alpha=0.7)
        
        # Add dynamic median lines
        g.ax_joint.axvline(rps_median, color='r', linestyle='--', label='Median RPS')
        g.ax_joint.axhline(latency_median, color='g', linestyle='--', label='Median Latency')
        
        # Add quadrant annotations
        g.ax_joint.text(0.4, 0.9, "High Perf", transform=g.ax_joint.transAxes,
                       fontsize=12, color='darkgreen')
        g.ax_joint.text(0.4, 0.1, "Underutilized", transform=g.ax_joint.transAxes,
                       fontsize=12, color='darkblue')
        g.ax_joint.text(0.05, 0.9, "Overloaded", transform=g.ax_joint.transAxes,
                       fontsize=12, color='darkred')
        
        plt.suptitle(title)
        return g
    
    # Predictive plot
    g_pred = create_quadrant_plot(predictive_df, "Predictive Performance Quadrant")
    plt.savefig(os.path.join(OUTPUT_DIR, "quadrant_predictive.png"))
    plt.close()
    
    # Reactive plot
    g_react = create_quadrant_plot(reactive_df, "Reactive Performance Quadrant")
    plt.savefig(os.path.join(OUTPUT_DIR, "quadrant_reactive.png"))
    plt.close()

# Additional insightful plots
def plot_traffic_pattern_comparison(predictive_df, reactive_df):
    """Compare surge patterns using minute-level aggregation"""
    plt.figure(figsize=(14, 8))
    
    # Create minute-based aggregation
    for df, name, color in zip([predictive_df, reactive_df], ['Predictive', 'Reactive'], ['blue', 'orange']):
        minute_avg = df.groupby('minute').agg({
            'requests_per_sec': 'mean',
            'pods': 'mean',
            'latency_avg_ms': 'mean'
        }).reset_index()
        
        plt.plot(minute_avg['minute'], minute_avg['requests_per_sec'],
                label=f'{name} Requests', color=color, linestyle='-')
        plt.plot(minute_avg['minute'], minute_avg['pods'],
                label=f'{name} Pods', color=color, linestyle='--')
    
    plt.title("Minute-level Traffic Pattern Comparison")
    plt.xlabel("Minute of Hour")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "minute_pattern_comparison.png"))
    plt.close()

def plot_latency_distribution_ridges(predictive_df, reactive_df):
    """Ridgeline plot of latency distributions"""
    plt.figure(figsize=(16, 12))
    
    # Combine data
    combined = pd.concat([
        predictive_df.assign(type='Predictive'),
        reactive_df.assign(type='Reactive')
    ])
    
    # Create hour-based facets
    pal = sns.color_palette("coolwarm", n_colors=combined['hour'].nunique())
    g = sns.FacetGrid(combined, row='hour', hue='hour', palette=pal, aspect=15, height=0.75)
    
    g.map(sns.kdeplot, 'p90_ms', bw_adjust=0.5, clip=[0, 500], fill=True, alpha=0.6)
    
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle("Hourly p90 Latency Distribution Comparison")
    plt.savefig(os.path.join(OUTPUT_DIR, "latency_ridges.png"))
    plt.close()

# Main execution
if __name__ == "__main__":
    predictive_df, reactive_df = load_data()
    
    # Generate required plots
    plot_latency_by_1hr_chunks_with_target_offset(predictive_df, reactive_df)
    plot_pods_with_target_rate(predictive_df, reactive_df)
    plot_requests_over_time(predictive_df, reactive_df)
    plot_target_vs_actual(predictive_df, reactive_df)
    plot_requests_per_pod(predictive_df, reactive_df)
    plot_performance_quadrant(predictive_df, reactive_df)
    
    # Generate additional insightful plots
    plot_traffic_pattern_comparison(predictive_df, reactive_df)
    plot_latency_distribution_ridges(predictive_df, reactive_df)