import csv
import json
import os
from datetime import datetime

# Ensure the output directory exists
os.makedirs('./json', exist_ok=True)

# Read CSV file
data = []
with open('./output/load_test_data.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        data.append(row)

# Format for 1-minute intervals
one_minute_data = []
for row in data:
    hour = int(row['hour'])
    minute = int(row['minute'])
    timestamp = f"{hour:02d}:{minute:02d}"
    pod_count = int(row['pods'])
    
    one_minute_data.append({
        "timestamp": timestamp,
        "podCount": pod_count
    })

# Format for 5-minute intervals
five_minute_data = []
seen_intervals = set()

for row in data:
    hour = int(row['hour'])
    minute = int(row['minute'])
    # Calculate the corresponding 5-minute interval
    interval_minute = (minute // 5) * 5
    interval_timestamp = f"{hour:02d}:{interval_minute:02d}"
    
    # Only add each 5-minute interval once (using the first occurrence)
    if interval_timestamp not in seen_intervals:
        seen_intervals.add(interval_timestamp)
        pod_count = int(row['pods'])
        
        five_minute_data.append({
            "timestamp": interval_timestamp,
            "podCount": pod_count
        })

# Save the 1-minute interval JSON
one_minute_json = {"data": one_minute_data}
with open('./json/traffic_1_interval.json', 'w') as jsonfile:
    json.dump(one_minute_json, jsonfile, indent=2)

# Save the 5-minute interval JSON
five_minute_json = {"data": five_minute_data}
with open('./json/traffic_5_interval.json', 'w') as jsonfile:
    json.dump(five_minute_json, jsonfile, indent=2)

print("JSON files created successfully:")
print("- ./json/traffic_1_interval.json")
print("- ./json/traffic_5_interval.json")