import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
from itertools import combinations
import hashlib
import os
import functools
import functools, hashlib, inspect, os, re, textwrap
from pathlib import Path

# Configuration
DATA_DIR = "./final_data"
OUTPUT_DIR = "./analysis_plots"
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "analysis_plots")).resolve()

def slugify(text: str, maxlen: int = 30) -> str:
    """Convert arbitrary text to a safe, compact directory fragment."""
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text[:maxlen] or "none"

def plot_cache(monitored_kwargs=None):
    """
    Parameters
    ----------
    monitored_kwargs : list[str] | None
        Names of kwargs that should appear in the directory slug.
        Defaults to *all* kwargs, sorted.
    """
    def decorator(func):
        # Pre-compute a hash of the *current* source code
        src_hash = hashlib.sha1(
            textwrap.dedent(inspect.getsource(func)).encode()
        ).hexdigest()[:7]

        @functools.wraps(func)
        def wrapper(*args, force=False, **kwargs):
            # Allow env-var override
            force = force or os.getenv("PLOT_CACHE_FORCE") == "1"

            # Build a readable parameter slug
            kw_to_use = monitored_kwargs or sorted(kwargs)
            pieces = [
                f"{k}{slugify(str(kwargs.get(k)))}" for k in kw_to_use if k in kwargs
            ] or ["default"]
            param_slug = "_".join(pieces)

            # Compose final output directory
            out_dir = OUTPUT_DIR / func.__name__ / f"{param_slug}_{src_hash}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Fast-path: skip if files already exist and no force flag
            if not force and any(out_dir.iterdir()):
                print(f"Skipping {func.__name__}; cached in {out_dir}")
                return out_dir

            # --- run the plotting code --------------------------------------
            prev_output_dir = os.environ.get("OUTPUT_DIR")
            os.environ["OUTPUT_DIR"] = str(out_dir)       # hand off to inner code
            try:
                func(*args, **kwargs)
            finally:
                # restore even if the plot function raised
                if prev_output_dir is None:
                    os.environ.pop("OUTPUT_DIR", None)
                else:
                    os.environ["OUTPUT_DIR"] = prev_output_dir
            # ----------------------------------------------------------------

            # Maintain a convenient 'latest' symlink
            latest_link = out_dir.parent / "latest"
            try:
                if latest_link.is_symlink() or latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(out_dir.name, target_is_directory=True)
            except OSError:
                # On Windows or restricted FS, silently ignore
                pass

            print(f"Generated {func.__name__} → {out_dir}")
            return out_dir

        return wrapper
    return decorator


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
def create_offset_windows(start_hour=4, start_minute=30, window_size=40):
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
        
        # Move window by dx minutes
        dx = 20
        current_hour += 0 if current_minute + 20 < 60 else 1
        current_minute = (current_minute + 20) % 60

    current_hour = start_hour
    current_minute = start_minute
    window_size += 20
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
        
        # Move window by dx minutes
        dx = 20
        current_hour += 0 if current_minute + 20 < 60 else 1
        current_minute = (current_minute + 20) % 60
    
    return windows

@plot_cache()
def plot_individual_latency(ax, chunk, title, start_time, end_time, latency_metrics, ymax=None):
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

    # Add axis limit control
    if ymax is not None:
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"Latency (ms, 0-{ymax})", color=colors[0])
    else:
        ax.set_ylabel("Latency (ms)", color=colors[0])

@plot_cache()
def plot_combined_latency(ax, p_chunk, r_chunk, start, end, latency_metrics, ymax=None):
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

    # Add axis limit control
    if ymax is not None:
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"Latency (ms, 0-{ymax})", color=colors[0])
    else:
        ax.set_ylabel("Latency (ms)", color=colors[0])

@plot_cache()
def plot_latency_by_1hr_chunks_with_target_offset(predictive_df, reactive_df, ymax=None):
    """Create latency analysis plots with controlled y-axis limits"""
    metric_groups = [
        ['p50_ms', 'p75_ms', 'p90_ms', 'p99_ms'],
        ['p50_ms', 'p75_ms', 'p90_ms', 'latency_avg_ms'],
        ['p50_ms', 'p75_ms'],
        ['p50_ms', 'latency_avg_ms']
    ]
    
    windows = create_offset_windows()
    
    # Create main directory for limited axis analysis
    analysis_dir = os.path.join(os.environ["OUTPUT_DIR"], "latency_analysis_ax_limited")
    os.makedirs(analysis_dir, exist_ok=True)
    
    for group_idx, latency_metrics in enumerate(metric_groups):
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
            
            # Combined plot with axis limit
            ax_combined = plt.subplot(gs[:, 0])
            plot_combined_latency(
                ax_combined, p_chunk, r_chunk, 
                start_time, end_time, latency_metrics,
                ymax=ymax
            )
            
            # Individual plots with axis limits
            ax_predictive = plt.subplot(gs[0, 1])
            plot_individual_latency(
                ax_predictive, p_chunk, "Predictive",
                start_time, end_time, latency_metrics,
                ymax=ymax
            )
            
            ax_reactive = plt.subplot(gs[1, 1])
            plot_individual_latency(
                ax_reactive, r_chunk, "Reactive",
                start_time, end_time, latency_metrics,
                ymax=ymax
            )
            
            # Save with time window identifier in metric group subdirectory
            filename = f"latency_{start_time.replace(':','')}-{end_time.replace(':','')}_ymax-{ymax}.png"
            plt.savefig(os.path.join(group_dir, filename))
            plt.close()

@plot_cache()
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
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "pods_target_comparison.png"))
    plt.close()

@plot_cache()
def plot_requests_over_time(predictive_df, reactive_df):
    """Visualize request patterns with multiple perspectives"""
    # Overlay plot
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=predictive_df, x='time_id', y='requests_per_sec',
                label='Predictive', color='blue')
    sns.lineplot(data=reactive_df, x='time_id', y='requests_per_sec',
                label='Reactive', color='orange')
    plt.title("Request Rate Comparison")
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "requests_overlay.png"))
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
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "requests_faceted.png"))
    plt.close()

@plot_cache()
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
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "target_vs_actual.png"))
    plt.close()

@plot_cache()
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
        plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], f"requests_per_pod_{name}.png"))
        plt.close()

@plot_cache()
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
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "quadrant_predictive.png"))
    plt.close()
    
    # Reactive plot
    g_react = create_quadrant_plot(reactive_df, "Reactive Performance Quadrant")
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "quadrant_reactive.png"))
    plt.close()

@plot_cache()
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
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "minute_pattern_comparison.png"))
    plt.close()

@plot_cache()
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
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "latency_ridges.png"))
    plt.close()

@plot_cache()
def plot_pods_over_time(predictive_df, reactive_df):
    """Visualize pod counts with comparative analysis"""
    # Overlay plot with target rates
    plt.figure(figsize=(18, 10))
    
    # Primary axis for pods
    ax1 = plt.gca()
    sns.lineplot(data=predictive_df, x='time_id', y='pods', 
                label='Predictive Pods', color='blue', ax=ax1)
    sns.lineplot(data=reactive_df, x='time_id', y='pods', 
                label='Reactive Pods', color='orange', ax=ax1)
    
    # Secondary axis for target rates
    ax2 = ax1.twinx()
    sns.lineplot(data=predictive_df, x='time_id', y='target_rate',
                label='Predictive Target', color='navy', linestyle='--', ax=ax2)
    sns.lineplot(data=reactive_df, x='time_id', y='target_rate',
                label='Reactive Target', color='darkorange', linestyle='--', ax=ax2)
    
    ax1.set_title("Pod Scaling vs Target Rates")
    ax1.set_ylabel("Pod Count")
    ax2.set_ylabel("Target Rate (req/s)")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "pods_target_overlay.png"))
    plt.close()

@plot_cache()
def plot_pod_efficiency(predictive_df, reactive_df):
    """Compare request handling efficiency per pod"""
    plt.figure(figsize=(15, 8))
    
    for df, color, name in [(predictive_df, 'blue', 'Predictive'), 
                           (reactive_df, 'orange', 'Reactive')]:
        df = df.copy()
        df['requests_per_pod'] = df['requests_per_sec'] / df['pods']
        sns.lineplot(data=df, x='time_id', y='requests_per_pod',
                    color=color, label=name)
    
    plt.title("Request Handling Efficiency per Pod Over Time")
    plt.ylabel("Requests/sec per Pod")
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "pod_efficiency.png"))
    plt.close()

@plot_cache()
def plot_pod_latency_correlation(predictive_df, reactive_df):
    """Analyze relationship between pod count and latency"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    for idx, (df, name) in enumerate(zip([predictive_df, reactive_df], 
                                       ['Predictive', 'Reactive'])):
        # Pods vs Latency
        sns.scatterplot(data=df, x='pods', y='latency_avg_ms', 
                       ax=axes[0][idx], color='blue' if idx==0 else 'orange')
        axes[0][idx].set_title(f"{name} System: Pods vs Avg Latency")
        
        # Pods vs Request Rate
        sns.scatterplot(data=df, x='pods', y='requests_per_sec',
                       ax=axes[1][idx], color='blue' if idx==0 else 'orange')
        axes[1][idx].set_title(f"{name} System: Pods vs Request Rate")
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "pod_correlations.png"))
    plt.close()

@plot_cache()
def analyze_time_windows(predictive_df, reactive_df):
    """Perform statistical analysis for each time window"""
    windows = create_offset_windows()
    analysis_dir = os.path.join(os.environ["OUTPUT_DIR"], "window_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # print(len(windows))
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
        
        stats = {
            'predictive': calculate_window_stats(p_chunk),
            'reactive': calculate_window_stats(r_chunk)
        }
        
        # Save statistics
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        filename = f"stats_{start_time.replace(':','')}-{end_time.replace(':','')}.csv"
        stats_df.to_csv(os.path.join(analysis_dir, filename))
        
        # Create visual comparison
        plot_window_comparison(stats, start_time, end_time, analysis_dir)

def calculate_window_stats(df):
    """Calculate comprehensive statistics for a window"""
    if df.empty:
        return {}
    
    return {
        'pod_count_avg': df['pods'].mean(),
        'pod_count_max': df['pods'].max(),
        'latency_avg': df['latency_avg_ms'].mean(),
        'p50_latency': df['p50_ms'].mean(),
        'p75_latency': df['p75_ms'].mean(),
        'p90_latency': df['p90_ms'].mean(),
        'p99_latency': df['p99_ms'].mean(),
        'request_rate_avg': df['requests_per_sec'].mean(),
        'target_rate_avg': df['target_rate'].mean(),
        'request_variation': df['req_sec_stdev_pct'].mean(),
        'throughput_mb': df['transfer_per_sec_mb'].sum()
    }

def plot_window_comparison(stats, start, end, output_dir):
    """Visualize comparative statistics for a window"""
    metrics = [
        ('pod_count_avg', 'Average Pod Count'),
        ('latency_avg', 'Average Latency (ms)'),
        ('request_rate_avg', 'Average Request Rate'),
        ('throughput_mb', 'Total Throughput (MB)')
    ]
    metrics2 = [
        ('p50_latency', 'P50 Latency (ms)'),
        ('p75_latency', 'P75 Latency (ms)'),
        ('p90_latency', 'P90 Latency (ms)'),
        ('p99_latency', 'P99 Latency (ms)'),
    ]
    
    plt.figure(figsize=(15, 10))
    for idx, (metric, title) in enumerate(metrics, 1):
        plt.subplot(3, 2, idx)
        values = [
            stats['predictive'].get(metric, 0),
            stats['reactive'].get(metric, 0)
        ]
        plt.bar(['Predictive', 'Reactive'], values, color=['blue', 'orange'])
        plt.title(title)
    
    plt.suptitle(f"Window {start}-{end} System Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], f"comparison_{start}-{end}.png"))
    print(f"saved under: {os.path.join(os.environ['OUTPUT_DIR'], 'comparison_{start}-{end}.png')}")
    plt.close()

    plt.figure(figsize=(15, 10))
    for idx, (metric, title) in enumerate(metrics2, 1):
        plt.subplot(3, 2, idx)
        values = [
            stats['predictive'].get(metric, 0),
            stats['reactive'].get(metric, 0)
        ]
        plt.bar(['Predictive', 'Reactive'], values, color=['blue', 'orange'])
        plt.title(title)
    
    plt.suptitle(f"Window {start}-{end} System Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], f"comparison_{start}-{end}_latency.png"))
    print(f"saved under: {os.path.join(os.environ['OUTPUT_DIR'], 'comparison_{start}-{end}.png')}")
    plt.close()

@plot_cache()
def plot_latency_and_pods_by_1hr_chunks(predictive_df, reactive_df, ymax=None):
    """Create latency analysis plots with pod count overlays"""
    metric_groups = [
        ['p50_ms', 'p90_ms', 'latency_avg_ms'],
        ['p75_ms', 'p99_ms'],
        ['latency_avg_ms', 'p99.9_ms']
    ]
    
    windows = create_offset_windows()
    
    analysis_dir = os.path.join(os.environ["OUTPUT_DIR"], "latency_pods_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    for group_idx, latency_metrics in enumerate(metric_groups):
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

            plt.figure(figsize=(22, 12))
            gs = GridSpec(2, 2, figure=plt.gcf())

            # Enhanced combined plot with pods
            ax_combined = plt.subplot(gs[:, 0])
            plot_enhanced_combined_latency(
                ax_combined, p_chunk, r_chunk,
                start_time, end_time, latency_metrics,
                ymax=ymax
            )

            # Individual plots with pod overlays
            ax_predictive = plt.subplot(gs[0, 1])
            plot_individual_latency_with_pods(
                ax_predictive, p_chunk, "Predictive",
                start_time, end_time, latency_metrics,
                ymax=ymax
            )

            ax_reactive = plt.subplot(gs[1, 1])
            plot_individual_latency_with_pods(
                ax_reactive, r_chunk, "Reactive",
                start_time, end_time, latency_metrics,
                ymax=ymax
            )

            filename = f"latency_pods_{start_time.replace(':','')}-{end_time.replace(':','')}_ymax-{ymax}.png"
            plt.savefig(os.path.join(group_dir, filename))
            plt.close()
    print(f"saved plot_latency_and_pods_by_1hr_chunks to {group_dir}")

def plot_individual_latency_with_pods(ax, chunk, title, start_time, end_time, 
                                     latency_metrics, ymax=None):
    """Enhanced individual plot with pod count overlay"""
    if chunk.empty:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        return

    # Plot latency metrics
    colors = sns.color_palette("husl", n_colors=len(latency_metrics))
    for idx, metric in enumerate(latency_metrics):
        sns.lineplot(data=chunk, x='time_id', y=metric, ax=ax,
                    label=f"{metric.replace('_ms', '')}", color=colors[idx])

    # Create twin axes
    ax2 = ax.twinx()  # For target rate and pods
    
    # Plot target rate
    ax2.plot(chunk['time_id'], chunk['target_rate'], 
            color='#2c3e50', linestyle='--', linewidth=1.5, 
            label='Target Rate', alpha=0.7)

    # Plot pods with filled area
    ax2.fill_between(chunk['time_id'], chunk['pods'], 
                    color='#27ae60', alpha=0.2, label='Pods')
    ax2.plot(chunk['time_id'], chunk['pods'], 
            color='#27ae60', linewidth=1.5, linestyle='-')

    # Axis configuration
    ax.set_title(f"{title} System: {start_time}-{end_time}\nLatency Metrics with Pod Scaling")
    ax.set_xlabel("Time (Decimal Hours)", fontsize=10)
    ax.set_ylabel("Latency (ms)", color=colors[0])
    ax2.set_ylabel("Target Rate (req/s) / Pod Count", color='#2c3e50')
    
    if ymax:
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"Latency (ms, 0-{ymax})", color=colors[0])

    # Legend handling
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    combined = [lines[0] + lines2[0], [lines[1] + lines2[1]]]
    ax.legend(*combined, loc='upper left', frameon=True,
             facecolor='white', framealpha=0.8)

def plot_enhanced_combined_latency(ax, p_chunk, r_chunk, start, end, 
                                  latency_metrics, ymax=None):
    """Enhanced combined plot with pod scaling visualization"""
    # Primary axis setup
    colors = sns.color_palette("tab10", n_colors=len(latency_metrics))
    
    # Plot latency metrics
    for idx, metric in enumerate(latency_metrics):
        if not p_chunk.empty:
            sns.lineplot(data=p_chunk, x='time_id', y=metric, ax=ax,
                        color=colors[idx], linestyle='-',
                        label=f'Predictive {metric.replace("_ms", "")}')
        if not r_chunk.empty:
            sns.lineplot(data=r_chunk, x='time_id', y=metric, ax=ax,
                        color=colors[idx], linestyle='--',
                        label=f'Reactive {metric.replace("_ms", "")}')

    # Twin axis for operational metrics
    ax2 = ax.twinx()
    
    # Plot pod counts
    if not p_chunk.empty:
        ax2.fill_between(p_chunk['time_id'], p_chunk['pods'],
                        color='#3498db', alpha=0.2, label='Predictive Pods')
    if not r_chunk.empty:
        ax2.fill_between(r_chunk['time_id'], r_chunk['pods'],
                        color='#e74c3c', alpha=0.2, label='Reactive Pods')

    # Configuration
    ax.set_title(f"Combined Analysis: {start}-{end}", fontsize=12)
    ax.set_xlabel("Time (Decimal Hours)", fontsize=10)
    ax.set_ylabel("Latency (ms)", color=colors[0])
    ax2.set_ylabel("Pod Count", color='#2c3e50')
    
    if ymax:
        ax.set_ylim(0, ymax)

    # Legend handling
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    combined = [lines[0] + lines2[0], [lines[1] + lines2[1]]]
    ax.legend(*combined, loc='upper left', frameon=True,
             facecolor='white', framealpha=0.8)

# Main execution
if __name__ == "__main__":
    predictive_df, reactive_df = load_data()
    
    # Generate required plots
    plot_latency_by_1hr_chunks_with_target_offset(predictive_df, reactive_df)
    plot_latency_by_1hr_chunks_with_target_offset(predictive_df, reactive_df, ymax=200)
    plot_latency_by_1hr_chunks_with_target_offset(predictive_df, reactive_df, ymax=600)
    plot_pods_with_target_rate(predictive_df, reactive_df)
    plot_requests_over_time(predictive_df, reactive_df)
    plot_target_vs_actual(predictive_df, reactive_df)
    plot_requests_per_pod(predictive_df, reactive_df)
    plot_performance_quadrant(predictive_df, reactive_df)

    # Generate all analyses
    plot_pods_over_time(predictive_df, reactive_df)
    plot_pod_efficiency(predictive_df, reactive_df)
    plot_pod_latency_correlation(predictive_df, reactive_df)
    analyze_time_windows(predictive_df, reactive_df)
    
    # Generate additional insightful plots
    plot_traffic_pattern_comparison(predictive_df, reactive_df)
    plot_latency_distribution_ridges(predictive_df, reactive_df)

    plot_latency_and_pods_by_1hr_chunks(predictive_df, reactive_df)
    plot_latency_and_pods_by_1hr_chunks(predictive_df, reactive_df, ymax=400)

