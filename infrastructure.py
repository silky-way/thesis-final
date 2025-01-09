import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% ROAD WIDTH -- CREATE THREE OVERVIEW PLOTS OF AVERAGE SPEED


# Define the fixed user mix distribution
def parse_scenario(filename):
    bike_pct = 0.3
    e_bike_pct = 0.4
    speedpedelec_pct = 0.3
    return bike_pct, e_bike_pct, speedpedelec_pct

# Load all scenario files
csv_files = glob.glob('infrastructure/infra*.csv')
results = []

# Process each scenario file
for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    
    # Calculate speed (magnitude of velocity)
    df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    
    # Calculate deviation (perpendicular to the road, e.g., position_y)
    df['deviation'] = df['position_y'].abs()  # Use absolute value for deviation
    
    # Calculate acceleration (magnitude of acceleration)
    df['acceleration'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
    
    # Calculate mean values per vehicle type
    mean_values = df.groupby('vehicle')[['speed', 'deviation', 'acceleration']].mean().reset_index()
    mean_values['scenario'] = file  # Add scenario identifier
    results.append(mean_values)

# Combine results into a single DataFrame
final_df = pd.concat(results, ignore_index=True)

# Extract scenario numbers for plotting
final_df['scenario_num'] = final_df['scenario'].str.extract(r'infra(\d+)').astype(int)

# Define colors for each vehicle type
color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Calculate global min and max for each metric with a buffer
def calculate_limits(data, metric, buffer_ratio=0.1):
    global_min = data[metric].min()
    global_max = data[metric].max()
    buffer = buffer_ratio * (global_max - global_min)
    y_min = max(0, global_min - buffer)  # Ensure y_min is not negative
    y_max = global_max + buffer
    return y_min, y_max

# Calculate y-axis limits for speed, deviation, and acceleration
speed_y_min, speed_y_max = calculate_limits(final_df, 'speed')
deviation_y_min, deviation_y_max = calculate_limits(final_df, 'deviation')
acceleration_y_min, acceleration_y_max = calculate_limits(final_df, 'acceleration')

# Create a figure with three subplots for each metric
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

# Plot for speed
for i, vehicle in enumerate(color_map.keys()):
    ax = axes[0, i]
    vehicle_data = final_df[final_df['vehicle'] == vehicle]
    ax.scatter(vehicle_data['scenario_num'], vehicle_data['speed'], color=color_map[vehicle], label=vehicle.capitalize())
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Mean Speed')
    ax.set_title(f'Mean Speed of {vehicle.capitalize()} Across Scenarios')
    ax.set_ylim(speed_y_min, speed_y_max)  # Set dynamic y-axis limits
    ax.grid(True)
    ax.legend()

# Plot for deviation
for i, vehicle in enumerate(color_map.keys()):
    ax = axes[1, i]
    vehicle_data = final_df[final_df['vehicle'] == vehicle]
    ax.scatter(vehicle_data['scenario_num'], vehicle_data['deviation'], color=color_map[vehicle], label=vehicle.capitalize())
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Mean Deviation')
    ax.set_title(f'Mean Deviation of {vehicle.capitalize()} Across Scenarios')
    ax.set_ylim(deviation_y_min, deviation_y_max)  # Set dynamic y-axis limits
    ax.grid(True)
    ax.legend()

# Plot for acceleration
for i, vehicle in enumerate(color_map.keys()):
    ax = axes[2, i]
    vehicle_data = final_df[final_df['vehicle'] == vehicle]
    ax.scatter(vehicle_data['scenario_num'], vehicle_data['acceleration'], color=color_map[vehicle], label=vehicle.capitalize())
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Mean Acceleration')
    ax.set_title(f'Mean Acceleration of {vehicle.capitalize()} Across Scenarios')
    ax.set_ylim(acceleration_y_min, acceleration_y_max)  # Set dynamic y-axis limits
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# %%
