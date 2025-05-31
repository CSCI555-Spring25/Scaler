

import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import sys
import numpy as np

def parse_logs(log_file):
    """Parse the local-logs.txt file to extract HPA changes."""
    
    # Regex patterns
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]'
    update_hpa_pattern = r'Updating HPA (\S+) to minReplicas=(\d+)'
    success_update_pattern = r'Successfully updated HPA (\S+) minReplicas to (\d+)'
    current_settings_pattern = r'Current HPA (\S+) settings: minReplicas=(\d+), maxReplicas=(\d+)'
    historical_update_pattern = r'Updated historical data for timestamp (\d{2}:\d{2}) from (\d+) to (\d+) \(historical: (\d+), current: (\d+)\)'
    
    # Data structures
    scaling_events = []
    current_settings = []
    historical_updates = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Extract timestamp
                timestamp_match = re.search(timestamp_pattern, line)
                if not timestamp_match:
                    continue
                
                timestamp = timestamp_match.group(1)
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                
                # Extract HPA update requests
                update_match = re.search(update_hpa_pattern, line)
                if update_match:
                    hpa_name = update_match.group(1)
                    min_replicas = int(update_match.group(2))
                    scaling_events.append({
                        'timestamp': dt,
                        'event_type': 'update_request',
                        'hpa_name': hpa_name,
                        'min_replicas': min_replicas
                    })
                    continue
                
                # Extract successful updates
                success_match = re.search(success_update_pattern, line)
                if success_match:
                    hpa_name = success_match.group(1)
                    min_replicas = int(success_match.group(2))
                    scaling_events.append({
                        'timestamp': dt,
                        'event_type': 'update_success',
                        'hpa_name': hpa_name,
                        'min_replicas': min_replicas
                    })
                    continue
                
                # Extract current settings
                settings_match = re.search(current_settings_pattern, line)
                if settings_match:
                    hpa_name = settings_match.group(1)
                    min_replicas = int(settings_match.group(2))
                    max_replicas = int(settings_match.group(3))
                    current_settings.append({
                        'timestamp': dt,
                        'hpa_name': hpa_name,
                        'min_replicas': min_replicas,
                        'max_replicas': max_replicas
                    })
                    continue
                
                # Extract historical data updates
                historical_match = re.search(historical_update_pattern, line)
                if historical_match:
                    time_str = historical_match.group(1)
                    from_value = int(historical_match.group(2))
                    to_value = int(historical_match.group(3))
                    historical_value = int(historical_match.group(4))
                    current_value = int(historical_match.group(5))
                    
                    # Create a time object for the historical timestamp
                    # Use the date from dt but time from the match
                    hour, minute = map(int, time_str.split(':'))
                    historical_dt = dt.replace(hour=hour, minute=minute)
                    
                    historical_updates.append({
                        'timestamp': dt,
                        'historical_timestamp': historical_dt,
                        'from_value': from_value,
                        'to_value': to_value,
                        'historical_value': historical_value,
                        'current_value': current_value
                    })
                    continue
                
        return {
            'scaling_events': scaling_events,
            'current_settings': current_settings,
            'historical_updates': historical_updates
        }
    
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return None

def analyze_hpa_changes(parsed_data):
    """Analyze the HPA changes for patterns and statistics."""
    
    if not parsed_data or 'scaling_events' not in parsed_data:
        print("No valid data to analyze")
        return None
    
    events = parsed_data['scaling_events']
    settings = parsed_data['current_settings']
    
    # Group by HPA name
    hpa_changes = defaultdict(list)
    for event in events:
        if event['event_type'] == 'update_success':
            hpa_changes[event['hpa_name']].append({
                'timestamp': event['timestamp'],
                'min_replicas': event['min_replicas']
            })
    
    # Calculate change statistics
    stats = {}
    for hpa_name, changes in hpa_changes.items():
        if len(changes) < 2:
            continue
            
        replica_values = [change['min_replicas'] for change in changes]
        
        # Calculate change deltas and prepare time series data
        deltas = []
        hourly_patterns = {}
        changes_by_hour = defaultdict(list)
        
        # Sort changes by timestamp
        sorted_changes = sorted(changes, key=lambda x: x['timestamp'])
        
        for i in range(1, len(sorted_changes)):
            prev = sorted_changes[i-1]
            curr = sorted_changes[i]
            delta = curr['min_replicas'] - prev['min_replicas']
            time_diff = (curr['timestamp'] - prev['timestamp']).total_seconds() / 60.0
            
            deltas.append({
                'timestamp': curr['timestamp'],
                'from': prev['min_replicas'],
                'to': curr['min_replicas'],
                'delta': delta,
                'time_diff_minutes': time_diff
            })
            
            # Group by hour for pattern analysis
            hour = curr['timestamp'].hour
            changes_by_hour[hour].append(delta)
        
        # Calculate hourly patterns
        for hour, hour_deltas in changes_by_hour.items():
            if len(hour_deltas) > 0:
                hourly_patterns[hour] = {
                    'avg_delta': sum(hour_deltas) / len(hour_deltas),
                    'max_delta': max(hour_deltas),
                    'min_delta': min(hour_deltas),
                    'count': len(hour_deltas)
                }
        
        # Find scale-up and scale-down patterns
        scale_ups = [d for d in deltas if d['delta'] > 0]
        scale_downs = [d for d in deltas if d['delta'] < 0]
        
        # Find periods of stability (no changes)
        stability_periods = []
        if len(sorted_changes) > 1:
            current_stable_value = sorted_changes[0]['min_replicas']
            stable_start = sorted_changes[0]['timestamp']
            
            for i in range(1, len(sorted_changes)):
                if sorted_changes[i]['min_replicas'] != current_stable_value:
                    # End of stability period
                    if (sorted_changes[i]['timestamp'] - stable_start).total_seconds() > 600:  # >10 min
                        stability_periods.append({
                            'start': stable_start,
                            'end': sorted_changes[i-1]['timestamp'],
                            'duration_minutes': (sorted_changes[i-1]['timestamp'] - stable_start).total_seconds() / 60,
                            'min_replicas': current_stable_value
                        })
                    # Start new stability period
                    current_stable_value = sorted_changes[i]['min_replicas']
                    stable_start = sorted_changes[i]['timestamp']
        
        stats[hpa_name] = {
            'min_replicas': min(replica_values),
            'max_replicas': max(replica_values),
            'avg_replicas': sum(replica_values) / len(replica_values),
            'num_changes': len(changes),
            'changes': changes,
            'deltas': deltas,
            'hourly_patterns': hourly_patterns,
            'scale_ups': scale_ups,
            'scale_downs': scale_downs,
            'stability_periods': stability_periods
        }
    
    return stats

def detect_daily_patterns(stats):
    """Detect hourly and daily patterns in scaling changes."""
    
    patterns = {}
    
    for hpa_name, hpa_stats in stats.items():
        hourly_data = hpa_stats['hourly_patterns']
        
        # Find peak scaling hours (most scale-up events)
        if hourly_data:
            scale_up_hours = []
            scale_down_hours = []
            
            for hour in range(24):
                if hour in hourly_data:
                    avg_delta = hourly_data[hour]['avg_delta']
                    if avg_delta > 0:
                        scale_up_hours.append((hour, avg_delta))
                    elif avg_delta < 0:
                        scale_down_hours.append((hour, avg_delta))
            
            # Sort by strength of scaling
            scale_up_hours.sort(key=lambda x: x[1], reverse=True)
            scale_down_hours.sort(key=lambda x: x[1])
            
            patterns[hpa_name] = {
                'primary_scale_up_hours': [h[0] for h in scale_up_hours[:3]] if scale_up_hours else [],
                'primary_scale_down_hours': [h[0] for h in scale_down_hours[:3]] if scale_down_hours else [],
                'hourly_pattern_strength': len(hourly_data.keys()) / 24.0,  # Measure of how many hours show distinct patterns
            }
            
            # Check for cyclical patterns
            changes = hpa_stats['changes']
            if len(changes) >= 24:  # At least a day's worth of data
                sorted_changes = sorted(changes, key=lambda x: x['timestamp'])
                
                # Create time series with hourly bins
                start_time = sorted_changes[0]['timestamp']
                end_time = sorted_changes[-1]['timestamp']
                
                # Calculate autocorrelation to find repeating patterns
                values = [change['min_replicas'] for change in sorted_changes]
                if len(values) > 48:  # Need enough data for meaningful autocorrelation
                    patterns[hpa_name]['has_daily_cycle'] = check_for_daily_cycle(values)
                else:
                    patterns[hpa_name]['has_daily_cycle'] = False
    
    return patterns

def check_for_daily_cycle(values, threshold=0.5):
    """Use autocorrelation to check for daily cycle patterns in replica values."""
    try:
        # Simple autocorrelation for 24 hour lag
        if len(values) <= 24:
            return False
            
        # Convert to numpy array
        series = np.array(values)
        
        # Calculate lag-1 autocorrelation (adjacent values similarity)
        n = len(series)
        lag1_autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
        
        # Calculate lag-24 autocorrelation (24-hour cycle similarity)
        if n <= 48:  # Not enough data for 24-hour lag
            return False
            
        lag24_autocorr = np.corrcoef(series[:-24], series[24:])[0, 1]
        
        # If lag-24 correlation is strong and stronger than lag-1, likely daily pattern
        return lag24_autocorr > threshold and lag24_autocorr > lag1_autocorr
    except:
        return False

def plot_hpa_changes(stats, parsed_data, output_file=None):
    """Generate visualizations of HPA changes over time."""
    
    if not stats:
        print("No statistics to plot")
        return
    
    # Create multiple plots - time series, hourly patterns, etc.
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))
    
    # Plot 1: minReplicas over time
    for hpa_name, hpa_stats in stats.items():
        changes = hpa_stats['changes']
        timestamps = [change['timestamp'] for change in changes]
        replicas = [change['min_replicas'] for change in changes]
        
        axs[0].plot(timestamps, replicas, marker='o', linestyle='-', label=hpa_name, linewidth=1.5)
    
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('minReplicas')
    axs[0].set_title('HPA minReplicas Changes Over Time')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Delta changes (increases/decreases)
    colors = {'simpleweb-hpa': 'blue'}  # Add more colors as needed
    
    for hpa_name, hpa_stats in stats.items():
        deltas = hpa_stats['deltas']
        if not deltas:
            continue
            
        timestamps = [d['timestamp'] for d in deltas]
        delta_values = [d['delta'] for d in deltas]
        
        # Use different colors for positive and negative deltas
        pos_timestamps = [timestamps[i] for i in range(len(timestamps)) if delta_values[i] > 0]
        pos_deltas = [delta_values[i] for i in range(len(delta_values)) if delta_values[i] > 0]
        
        neg_timestamps = [timestamps[i] for i in range(len(timestamps)) if delta_values[i] < 0]
        neg_deltas = [delta_values[i] for i in range(len(delta_values)) if delta_values[i] < 0]
        
        color = colors.get(hpa_name, 'blue')
        axs[1].scatter(pos_timestamps, pos_deltas, color='green', marker='^', label=f'{hpa_name} (Increase)', alpha=0.7)
        axs[1].scatter(neg_timestamps, neg_deltas, color='red', marker='v', label=f'{hpa_name} (Decrease)', alpha=0.7)
    
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Change in minReplicas')
    axs[1].set_title('HPA Scaling Events: Magnitude and Direction')
    axs[1].grid(True, alpha=0.3)
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Only show the legend for the first few items to avoid clutter
    handles, labels = axs[1].get_legend_handles_labels()
    if len(handles) > 4:
        axs[1].legend(handles[:4], labels[:4], loc='lower right')
    else:
        axs[1].legend(loc='lower right')
    
    # Plot 3: Historical data updates
    if 'historical_updates' in parsed_data and parsed_data['historical_updates']:
        historical_updates = parsed_data['historical_updates']
        
        # Sort by historical timestamp
        historical_updates.sort(key=lambda x: x['historical_timestamp'])
        
        # Extract data for plotting
        timestamps = [update['historical_timestamp'] for update in historical_updates]
        from_values = [update['from_value'] for update in historical_updates]
        to_values = [update['to_value'] for update in historical_updates]
        
        # Plot old values and updated values as separate lines
        axs[2].plot(timestamps, from_values, 'o', color='blue', alpha=0.7, label='Old Value')
        axs[2].plot(timestamps, to_values, 'o', color='green', alpha=0.7, label='Updated Value')
        
        # Draw vertical lines between old and new values at each timestamp
        for i in range(len(timestamps)):
            if from_values[i] != to_values[i]:
                # Determine color based on whether value increased or decreased
                color = 'green' if to_values[i] > from_values[i] else 'red'
                axs[2].plot([timestamps[i], timestamps[i]], [from_values[i], to_values[i]], 
                           color=color, linestyle='-', linewidth=2, alpha=0.7)
        
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Replica Count')
        axs[2].set_title('Historical Data Updates')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        axs[2].tick_params(axis='x', rotation=45)
    else:
        axs[2].set_title('No Historical Data Updates Found')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        # Create additional plots
        plt.figure(figsize=(12, 6))
        
        # Plot hourly patterns
        hours = list(range(24))
        hourly_avg_deltas = [0] * 24
        
        for hpa_name, hpa_stats in stats.items():
            hourly_patterns = hpa_stats['hourly_patterns']
            
            for hour, data in hourly_patterns.items():
                hourly_avg_deltas[hour] = data['avg_delta']
        
        bars = plt.bar(hours, hourly_avg_deltas, alpha=0.7)
        
        # Color bars based on positive/negative values
        for i, bar in enumerate(bars):
            if hourly_avg_deltas[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Change in minReplicas')
        plt.title('Hourly Scaling Patterns')
        plt.xticks(hours)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        hourly_output = output_file.replace('.png', '_hourly.png')
        plt.savefig(hourly_output, bbox_inches='tight')
        plt.close()
        
        # Create a separate plot for historical data updates if available
        if 'historical_updates' in parsed_data and parsed_data['historical_updates']:
            plt.figure(figsize=(14, 8))
            
            historical_updates = parsed_data['historical_updates']
            historical_updates.sort(key=lambda x: x['historical_timestamp'])
            
            timestamps = [update['historical_timestamp'] for update in historical_updates]
            from_values = [update['from_value'] for update in historical_updates]
            to_values = [update['to_value'] for update in historical_updates]
            
            # Plot old values and updated values as separate points
            plt.plot(timestamps, from_values, 'o', color='blue', alpha=0.7, label='Old Value')
            plt.plot(timestamps, to_values, 'o', color='green', alpha=0.7, label='Updated Value')
            
            # Draw vertical lines between old and new values at each timestamp
            for i in range(len(timestamps)):
                if from_values[i] != to_values[i]:
                    # Determine color based on whether value increased or decreased
                    color = 'green' if to_values[i] > from_values[i] else 'red'
                    plt.plot([timestamps[i], timestamps[i]], [from_values[i], to_values[i]], 
                             color=color, linestyle='-', linewidth=2, alpha=0.7)
                    
                    # Add text annotation showing the change
                    plt.annotate(f"{from_values[i]} → {to_values[i]}", 
                        xy=(timestamps[i], (from_values[i] + to_values[i])/2),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=8,
                        ha='left',
                        va='center')
            
            plt.xlabel('Time')
            plt.ylabel('Historical Replica Count')
            plt.title('Historical Data Updates Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            historical_output = output_file.replace('.png', '_historical.png')
            plt.savefig(historical_output, bbox_inches='tight')
            plt.close()
    else:
        plt.show()

def generate_report(parsed_data, stats, pattern_data=None):
    """Generate a text report summarizing the HPA changes."""
    
    report = []
    report.append("=" * 60)
    report.append("HPA Scaling Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    if not stats:
        report.append("No HPA changes found in the logs")
        return "\n".join(report)
    
    # Summary section
    report.append("Summary of HPA Changes:")
    report.append("-" * 40)
    for hpa_name, hpa_stats in stats.items():
        report.append(f"HPA: {hpa_name}")
        report.append(f"  Number of scaling events: {hpa_stats['num_changes']}")
        report.append(f"  Min replicas setting: {hpa_stats['min_replicas']}")
        report.append(f"  Max replicas setting: {hpa_stats['max_replicas']}")
        report.append(f"  Average replicas: {hpa_stats['avg_replicas']:.2f}")
        
        # Scale up/down stats
        scale_ups = hpa_stats['scale_ups']
        scale_downs = hpa_stats['scale_downs']
        report.append(f"  Scale-up events: {len(scale_ups)}")
        report.append(f"  Scale-down events: {len(scale_downs)}")
        
        if pattern_data and hpa_name in pattern_data:
            patterns = pattern_data[hpa_name]
            report.append(f"  Peak scale-up hours: {[f'{h:02d}:00' for h in patterns['primary_scale_up_hours']]}")
            report.append(f"  Peak scale-down hours: {[f'{h:02d}:00' for h in patterns['primary_scale_down_hours']]}")
            report.append(f"  Daily pattern detected: {'Yes' if patterns.get('has_daily_cycle', False) else 'No'}")
        
        report.append("")
    
    # Historical data updates section
    if 'historical_updates' in parsed_data and parsed_data['historical_updates']:
        report.append("\nHistorical Data Updates:")
        report.append("-" * 40)
        
        historical_updates = sorted(parsed_data['historical_updates'], key=lambda x: x['historical_timestamp'])
        
        for update in historical_updates:
            timestamp = update['historical_timestamp'].strftime('%Y-%m-%d %H:%M')
            from_val = update['from_value']
            to_val = update['to_value']
            historical = update['historical_value']
            current = update['current_value']
            
            change_str = f"{from_val} → {to_val}"
            if from_val != to_val:
                delta = to_val - from_val
                if delta > 0:
                    change_str += f" (+{delta})"
                else:
                    change_str += f" ({delta})"
            
            report.append(f"  {timestamp}: {change_str} (historical: {historical}, current: {current})")
        
        report.append("")
    
    # Detailed analysis section
    report.append("\nDetailed Analysis:")
    report.append("-" * 40)
    for hpa_name, hpa_stats in stats.items():
        report.append(f"HPA: {hpa_name}")
        
        # Stability periods
        stability_periods = hpa_stats['stability_periods']
        if stability_periods:
            report.append(f"  Stability periods (> 10 minutes at same replica count):")
            for i, period in enumerate(stability_periods[:10]):  # Show top 10
                start = period['start'].strftime('%Y-%m-%d %H:%M:%S')
                end = period['end'].strftime('%Y-%m-%d %H:%M:%S')
                duration = period['duration_minutes']
                report.append(f"    Period {i+1}: {start} to {end} ({duration:.1f} minutes) at {period['min_replicas']} replicas")
            
            if len(stability_periods) > 10:
                report.append(f"    ...and {len(stability_periods) - 10} more periods")
            
            report.append("")
        
        # Largest scale-up events
        if scale_ups:
            largest_ups = sorted(scale_ups, key=lambda x: x['delta'], reverse=True)[:5]
            report.append(f"  Largest scale-up events:")
            for i, event in enumerate(largest_ups):
                timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"    {i+1}. {timestamp}: {event['from']} → {event['to']} (+{event['delta']})")
            report.append("")
        
        # Largest scale-down events
        if scale_downs:
            largest_downs = sorted(scale_downs, key=lambda x: x['delta'])[:5]
            report.append(f"  Largest scale-down events:")
            for i, event in enumerate(largest_downs):
                timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"    {i+1}. {timestamp}: {event['from']} → {event['to']} ({event['delta']})")
            report.append("")
        
        # Hourly patterns
        hourly_patterns = hpa_stats['hourly_patterns']
        if hourly_patterns:
            active_hours = sorted(hourly_patterns.keys())
            report.append(f"  Hourly scaling patterns:")
            for hour in active_hours:
                pattern = hourly_patterns[hour]
                direction = "up" if pattern['avg_delta'] > 0 else "down"
                report.append(f"    {hour:02d}:00 - {hour+1:02d}:00: Average {direction} by {abs(pattern['avg_delta']):.2f} replicas ({pattern['count']} events)")
            report.append("")
        
        report.append("")
    
    # Detailed timeline section
    report.append("\nDetailed Timeline of Changes:")
    report.append("-" * 40)
    for hpa_name, hpa_stats in stats.items():
        report.append(f"HPA: {hpa_name}")
        changes = sorted(hpa_stats['changes'], key=lambda x: x['timestamp'])
        
        for i, change in enumerate(changes):
            timestamp = change['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            replicas = change['min_replicas']
            
            # Calculate change from previous setting if applicable
            change_str = ""
            if i > 0:
                prev_replicas = changes[i-1]['min_replicas']
                diff = replicas - prev_replicas
                if diff > 0:
                    change_str = f" (+{diff})"
                elif diff < 0:
                    change_str = f" ({diff})"
            
            report.append(f"  {timestamp}: minReplicas = {replicas}{change_str}")
        
        report.append("")
    
    return "\n".join(report)

def main():
    if len(sys.argv) < 2:
        print("Usage: python log_analyzer.py <log_file_path> [output_report_path] [output_plot_path]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    report_file = sys.argv[2] if len(sys.argv) > 2 else "hpa_scaling_report.txt"
    plot_file = sys.argv[3] if len(sys.argv) > 3 else "hpa_scaling_plot.png"
    
    # Parse logs
    print(f"Parsing log file: {log_file}")
    parsed_data = parse_logs(log_file)
    
    if not parsed_data:
        print("Failed to parse log file or no data found")
        sys.exit(1)
    
    # Analyze changes
    print("Analyzing HPA changes...")
    stats = analyze_hpa_changes(parsed_data)
    
    # Detect patterns
    print("Detecting temporal patterns...")
    patterns = detect_daily_patterns(stats)
    
    # Generate report
    print(f"Generating report to: {report_file}")
    report = generate_report(parsed_data, stats, patterns)
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Generate plot
    print(f"Generating plots to: {plot_file}")
    plot_hpa_changes(stats, parsed_data, plot_file)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 