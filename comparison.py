import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% CREATE THREE OVERVIEW PLOTS OF AVERAGE SPEED

def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    mean_speeds = df.groupby('vehicle')['speed'].mean().reset_index()
    mean_speeds['bike_pct'] = bike_pct
    mean_speeds['e_bike_pct'] = e_bike_pct
    mean_speeds['speedpedelec_pct'] = speedpedelec_pct
    results.append(mean_speeds)

final_df = pd.concat(results, ignore_index=True)

final_df['scenario'] = final_df['bike_pct'].astype(str) + '_' + final_df['e_bike_pct'].astype(str) + '_' + final_df['speedpedelec_pct'].astype(str)

pivot_df = final_df.pivot_table(index='scenario', columns='vehicle', values='speed')

# Correctly extract percentages from the scenario string
pivot_df['bike_pct'] = pivot_df.index.str.split('_').str[0].astype(int)
pivot_df['e_bike_pct'] = pivot_df.index.str.split('_').str[1].astype(int)
pivot_df['speedpedelec_pct'] = pivot_df.index.str.split('_').str[2].astype(int)

# Define vehicle types and their corresponding percentage columns
vehicle_info = [
    ('bike', 'bike_pct'),
    ('ebike', 'e_bike_pct'),
    ('speed_pedelec', 'speedpedelec_pct')
]

# Define color mapping
color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Create a figure with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for i, (vehicle, pct_col) in enumerate(vehicle_info):
    ax = axes[i]
    # Filter data for the current vehicle
    data = pivot_df[[vehicle, pct_col]].dropna()
    # Sort by the percentage column
    data = data.sort_values(by=pct_col)
    # Plot with specified color
    ax.plot(data[pct_col], data[vehicle], marker='o', color=color_map[vehicle])
    ax.set_xlabel(f'{vehicle} (%)')
    ax.set_ylabel('Mean Speed')
    ax.set_title(f'Mean Speed of {vehicle} over all scenarios')
    ax.grid(True)

plt.tight_layout()
plt.show()

#%% CREATE THREE PLOTS WITH TRENDLINE FOR AVERAGE SPEED

def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    mean_speeds = df.groupby('vehicle')['speed'].mean().reset_index()
    mean_speeds['bike_pct'] = bike_pct
    mean_speeds['e_bike_pct'] = e_bike_pct
    mean_speeds['speedpedelec_pct'] = speedpedelec_pct
    results.append(mean_speeds)

final_df = pd.concat(results, ignore_index=True)

# For bike plot: group by bike_pct
bike_group = final_df[final_df['vehicle'] == 'bike'].groupby('bike_pct')['speed'].mean().reset_index()

# For ebike plot: group by e_bike_pct
ebike_group = final_df[final_df['vehicle'] == 'ebike'].groupby('e_bike_pct')['speed'].mean().reset_index()

# For speed_pedelec plot: group by speedpedelec_pct
speedpedelec_group = final_df[final_df['vehicle'] == 'speed_pedelec'].groupby('speedpedelec_pct')['speed'].mean().reset_index()

y_data = []
vehicle_info = [
    ('bike', bike_group['speed'].tolist()),
    ('ebike', ebike_group['speed'].tolist()),
    ('speed_pedelec', speedpedelec_group['speed'].tolist())
]

for _, speeds in vehicle_info:
    y_data.extend(speeds)

global_min = np.nanmin(y_data) - 0.5
global_max = np.nanmax(y_data) + 0.5

color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot for bike
ax = axes[0]
x = bike_group['bike_pct']
y = bike_group['speed']
ax.scatter(x, y, color=color_map['bike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Bike (%)')
ax.set_ylabel('Mean Speed')
ax.set_title('Trendline Mean Speed of Bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for ebike
ax = axes[1]
x = ebike_group['e_bike_pct']
y = ebike_group['speed']
ax.scatter(x, y, color=color_map['ebike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('E-bike (%)')
ax.set_ylabel('Mean Speed')
ax.set_title('Trendline Mean Speed of E-bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for speed_pedelec
ax = axes[2]
x = speedpedelec_group['speedpedelec_pct']
y = speedpedelec_group['speed']
ax.scatter(x, y, color=color_map['speed_pedelec'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Speed Pedelec (%)')
ax.set_ylabel('Mean Speed')
ax.set_title('Trendline Mean Speed of Speed Pedelec over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

#%% CREATE THREE OVERVIEW PLOTS OF Y-DEVIATION

def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    # Calculate y deviation
    y_deviation = df.groupby('vehicle')['velocity_y'].std().reset_index()
    y_deviation.rename(columns={'velocity_y': 'y_deviation'}, inplace=True)
    y_deviation['bike_pct'] = bike_pct
    y_deviation['e_bike_pct'] = e_bike_pct
    y_deviation['speedpedelec_pct'] = speedpedelec_pct
    results.append(y_deviation)

final_df = pd.concat(results, ignore_index=True)

final_df['scenario'] = final_df['bike_pct'].astype(str) + '_' + final_df['e_bike_pct'].astype(str) + '_' + final_df['speedpedelec_pct'].astype(str)

pivot_df = final_df.pivot_table(index='scenario', columns='vehicle', values='y_deviation')

# Correctly extract percentages from the scenario string
pivot_df['bike_pct'] = pivot_df.index.str.split('_').str[0].astype(int)
pivot_df['e_bike_pct'] = pivot_df.index.str.split('_').str[1].astype(int)
pivot_df['speedpedelec_pct'] = pivot_df.index.str.split('_').str[2].astype(int)

# Define vehicle types and their corresponding percentage columns
vehicle_info = [
    ('bike', 'bike_pct'),
    ('ebike', 'e_bike_pct'),
    ('speed_pedelec', 'speedpedelec_pct')
]

# Define color mapping
color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Create a figure with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for i, (vehicle, pct_col) in enumerate(vehicle_info):
    ax = axes[i]
    # Filter data for the current vehicle
    data = pivot_df[[vehicle, pct_col]].dropna()
    # Sort by the percentage column
    data = data.sort_values(by=pct_col)
    # Plot with specified color
    ax.plot(data[pct_col], data[vehicle], marker='o', color=color_map[vehicle])
    ax.set_xlabel(f'{vehicle} (%)')
    ax.set_ylabel('Y Deviation')
    ax.set_title(f'Y Deviation of {vehicle} over all scenarios')
    ax.grid(True)

plt.tight_layout()
plt.show()

#%% CREATE THREE PLOTS WITH TRENDLINE FOR Y-DEVIATION

def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    df['y_deviation'] = df['position_y']  # Use position_y as y deviation
    mean_y_dev = df.groupby('vehicle')['y_deviation'].mean().reset_index()
    mean_y_dev['bike_pct'] = bike_pct
    mean_y_dev['e_bike_pct'] = e_bike_pct
    mean_y_dev['speedpedelec_pct'] = speedpedelec_pct
    results.append(mean_y_dev)

final_df = pd.concat(results, ignore_index=True)

# For bike plot: group by bike_pct
bike_group = final_df[final_df['vehicle'] == 'bike'].groupby('bike_pct')['y_deviation'].mean().reset_index()

# For ebike plot: group by e_bike_pct
ebike_group = final_df[final_df['vehicle'] == 'ebike'].groupby('e_bike_pct')['y_deviation'].mean().reset_index()

# For speed_pedelec plot: group by speedpedelec_pct
speedpedelec_group = final_df[final_df['vehicle'] == 'speed_pedelec'].groupby('speedpedelec_pct')['y_deviation'].mean().reset_index()

y_data = []
vehicle_info = [
    ('bike', bike_group['y_deviation'].tolist()),
    ('ebike', ebike_group['y_deviation'].tolist()),
    ('speed_pedelec', speedpedelec_group['y_deviation'].tolist())
]

for _, deviations in vehicle_info:
    y_data.extend(deviations)

global_min = np.nanmin(y_data) - 0.5
global_max = np.nanmax(y_data) + 0.5

color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot for bike
ax = axes[0]
x = bike_group['bike_pct']
y = bike_group['y_deviation']
ax.scatter(x, y, color=color_map['bike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Bike (%)')
ax.set_ylabel('Mean Y Deviation')
ax.set_title('Trendline Mean Y Deviation of Bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for ebike
ax = axes[1]
x = ebike_group['e_bike_pct']
y = ebike_group['y_deviation']
ax.scatter(x, y, color=color_map['ebike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('E-bike (%)')
ax.set_ylabel('Mean Y Deviation')
ax.set_title('Trendline Mean Y Deviation of E-bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for speed_pedelec
ax = axes[2]
x = speedpedelec_group['speedpedelec_pct']
y = speedpedelec_group['y_deviation']
ax.scatter(x, y, color=color_map['speed_pedelec'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Speed Pedelec (%)')
ax.set_ylabel('Mean Y Deviation')
ax.set_title('Trendline Mean Y Deviation of Speed Pedelec over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


#%% CREATE THREE OVERVIEW PLOTS OF ACCELERATION

def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    df['acceleration'] = df['acceleration_x']
    mean_accels = df.groupby('vehicle')['acceleration'].mean().reset_index()
    mean_accels['bike_pct'] = bike_pct
    mean_accels['e_bike_pct'] = e_bike_pct
    mean_accels['speedpedelec_pct'] = speedpedelec_pct
    results.append(mean_accels)

final_df = pd.concat(results, ignore_index=True)

final_df['scenario'] = final_df['bike_pct'].astype(str) + '_' + final_df['e_bike_pct'].astype(str) + '_' + final_df['speedpedelec_pct'].astype(str)

pivot_df = final_df.pivot_table(index='scenario', columns='vehicle', values='acceleration')

# Correctly extract percentages from the scenario string
pivot_df['bike_pct'] = pivot_df.index.str.split('_').str[0].astype(int)
pivot_df['e_bike_pct'] = pivot_df.index.str.split('_').str[1].astype(int)
pivot_df['speedpedelec_pct'] = pivot_df.index.str.split('_').str[2].astype(int)

# Define vehicle types and their corresponding percentage columns
vehicle_info = [
    ('bike', 'bike_pct'),
    ('ebike', 'e_bike_pct'),
    ('speed_pedelec', 'speedpedelec_pct')
]

# Define color mapping
color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Create a figure with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for i, (vehicle, pct_col) in enumerate(vehicle_info):
    ax = axes[i]
    # Filter data for the current vehicle
    data = pivot_df[[vehicle, pct_col]].dropna()
    # Sort by the percentage column
    data = data.sort_values(by=pct_col)
    # Plot with specified color
    ax.plot(data[pct_col], data[vehicle], marker='o', color=color_map[vehicle])
    ax.set_xlabel(f'{vehicle} (%)')
    ax.set_ylabel('Mean Acceleration (m/s²)')
    ax.set_title(f'Mean Acceleration of {vehicle} over all scenarios')
    ax.grid(True)

plt.tight_layout()
plt.show()

#%% CREATE THREE PLOTS WITH TRENDLINE FOR ACCELERATION
def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    df['acceleration'] = df['acceleration_x']
    mean_accel = df.groupby('vehicle')['acceleration'].mean().reset_index()
    mean_accel['bike_pct'] = bike_pct
    mean_accel['e_bike_pct'] = e_bike_pct
    mean_accel['speedpedelec_pct'] = speedpedelec_pct
    results.append(mean_accel)

final_df = pd.concat(results, ignore_index=True)

# For bike plot: group by bike_pct
bike_group = final_df[final_df['vehicle'] == 'bike'].groupby('bike_pct')['acceleration'].mean().reset_index()

# For ebike plot: group by e_bike_pct
ebike_group = final_df[final_df['vehicle'] == 'ebike'].groupby('e_bike_pct')['acceleration'].mean().reset_index()

# For speed_pedelec plot: group by speedpedelec_pct
speedpedelec_group = final_df[final_df['vehicle'] == 'speed_pedelec'].groupby('speedpedelec_pct')['acceleration'].mean().reset_index()

y_data = []
vehicle_info = [
    ('bike', bike_group['acceleration'].tolist()),
    ('ebike', ebike_group['acceleration'].tolist()),
    ('speed_pedelec', speedpedelec_group['acceleration'].tolist())
]

for _, accels in vehicle_info:
    y_data.extend(accels)

global_min = np.nanmin(y_data) - 0.5
global_max = np.nanmax(y_data) + 0.5

color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot for bike
ax = axes[0]
x = bike_group['bike_pct']
y = bike_group['acceleration']
ax.scatter(x, y, color=color_map['bike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Bike (%)')
ax.set_ylabel('Mean Acceleration')
ax.set_title('Trendline Mean Acceleration of Bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for ebike
ax = axes[1]
x = ebike_group['e_bike_pct']
y = ebike_group['acceleration']
ax.scatter(x, y, color=color_map['ebike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('E-bike (%)')
ax.set_ylabel('Mean Acceleration')
ax.set_title('Trendline Mean Acceleration of E-bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for speed_pedelec
ax = axes[2]
x = speedpedelec_group['speedpedelec_pct']
y = speedpedelec_group['acceleration']
ax.scatter(x, y, color=color_map['speed_pedelec'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Speed Pedelec (%)')
ax.set_ylabel('Mean Acceleration')
ax.set_title('Trendline Mean Acceleration of Speed Pedelec over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


#%% CREATE THREE OVERVIEW PLOTS OF HEAVY BRAKING

def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    df['directional_acceleration'] = df['acceleration_x'] 
    heavy_braking_df = df[df['directional_acceleration'] <= -2]
    if not heavy_braking_df.empty:
        mean_accels = heavy_braking_df.groupby('vehicle')['directional_acceleration'].mean().reset_index()
        mean_accels['bike_pct'] = bike_pct
        mean_accels['e_bike_pct'] = e_bike_pct
        mean_accels['speedpedelec_pct'] = speedpedelec_pct
        results.append(mean_accels)

final_df = pd.concat(results, ignore_index=True)

final_df['scenario'] = final_df['bike_pct'].astype(str) + '_' + final_df['e_bike_pct'].astype(str) + '_' + final_df['speedpedelec_pct'].astype(str)

pivot_df = final_df.pivot_table(index='scenario', columns='vehicle', values='directional_acceleration')

pivot_df['bike_pct'] = pivot_df.index.str.split('_').str[0].astype(int)
pivot_df['e_bike_pct'] = pivot_df.index.str.split('_').str[1].astype(int)
pivot_df['speedpedelec_pct'] = pivot_df.index.str.split('_').str[2].astype(int)

vehicle_info = [
    ('bike', 'bike_pct'),
    ('ebike', 'e_bike_pct'),
    ('speed_pedelec', 'speedpedelec_pct')
]

color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for i, (vehicle, pct_col) in enumerate(vehicle_info):
    ax = axes[i]
    data = pivot_df[[vehicle, pct_col]].dropna()
    data = data.sort_values(by=pct_col)
    ax.plot(data[pct_col], data[vehicle], marker='o', color=color_map[vehicle])
    ax.set_xlabel(f'{vehicle} (%)')
    ax.set_ylabel('Mean Heavy Braking Acceleration (m/s²)')
    ax.set_title(f'Mean Heavy Braking Acceleration of {vehicle} over all scenarios')
    ax.grid(True)
    ax.axhline(y=-2, color='r', linestyle='--', label='Braking Threshold (-2 m/s²)')
    ax.legend()

plt.tight_layout()
plt.show()

#%% CREATE THREE PLOTS WITH TRENDLINE FOR HEAVY BRAKING
def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)

    df['directional_acceleration'] = df['acceleration_x']
    heavy_braking_df = df[df['directional_acceleration'] <= -2]
    if not heavy_braking_df.empty:
        mean_accel = heavy_braking_df.groupby('vehicle')['directional_acceleration'].mean().reset_index()
        mean_accel['bike_pct'] = bike_pct
        mean_accel['e_bike_pct'] = e_bike_pct
        mean_accel['speedpedelec_pct'] = speedpedelec_pct
        results.append(mean_accel)

final_df = pd.concat(results, ignore_index=True)

# For bike plot: group by bike_pct
bike_group = final_df[final_df['vehicle'] == 'bike'].groupby('bike_pct')['directional_acceleration'].mean().reset_index()

# For ebike plot: group by e_bike_pct
ebike_group = final_df[final_df['vehicle'] == 'ebike'].groupby('e_bike_pct')['directional_acceleration'].mean().reset_index()

# For speed_pedelec plot: group by speedpedelec_pct
speedpedelec_group = final_df[final_df['vehicle'] == 'speed_pedelec'].groupby('speedpedelec_pct')['directional_acceleration'].mean().reset_index()

y_data = []
vehicle_info = [
    ('bike', bike_group['directional_acceleration'].tolist()),
    ('ebike', ebike_group['directional_acceleration'].tolist()),
    ('speed_pedelec', speedpedelec_group['directional_acceleration'].tolist())
]

for _, accels in vehicle_info:
    y_data.extend(accels)

global_min = np.nanmin(y_data) - 0.5
global_max = np.nanmax(y_data) + 0.5

color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot for bike
ax = axes[0]
x = bike_group['bike_pct']
y = bike_group['directional_acceleration']
ax.scatter(x, y, color=color_map['bike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Bike (%)')
ax.set_ylabel('Mean Heavy Braking Acceleration (m/s²)')
ax.set_title('Trendline Mean Heavy Braking Acceleration of Bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for ebike
ax = axes[1]
x = ebike_group['e_bike_pct']
y = ebike_group['directional_acceleration']
ax.scatter(x, y, color=color_map['ebike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('E-bike (%)')
ax.set_ylabel('Mean Heavy Braking Acceleration (m/s²)')
ax.set_title('Trendline Mean Heavy Braking Acceleration of E-bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for speed_pedelec
ax = axes[2]
x = speedpedelec_group['speedpedelec_pct']
y = speedpedelec_group['directional_acceleration']
ax.scatter(x, y, color=color_map['speed_pedelec'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Speed Pedelec (%)')
ax.set_ylabel('Mean Heavy Braking Acceleration (m/s²)')
ax.set_title('Trendline Mean Heavy Braking Acceleration of Speed Pedelec over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()





# %% CREATE THREE PLOTS OF CONVOLUTION

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Function to parse scenario filename
def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

# Define grid parameters
x_min, x_max = -10, 10
y_min, y_max = -10, 10
resolution = 0.1
x = np.arange(x_min, x_max, resolution)
y = np.arange(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# Define Gaussian kernel function
def gaussian_kernel(pos, center, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-np.sum((pos - center)**2, axis=1) / (2 * sigma**2))

# Function to calculate maximum density
def calculate_max_density(group, X, Y, sigma):
    positions = group[['position_x', 'position_y']].values
    grid_points = np.dstack((X, Y)).reshape(-1, 2)
    density = np.sum(gaussian_kernel(grid_points[:, np.newaxis, :], positions, sigma), axis=1)
    density = density.reshape(X.shape)
    density = density**4  # Apply the fourth power as per the requirement
    max_density = np.max(density)
    return max_density

# Process each CSV file
csv_files = glob.glob('logs/*.csv')
results = []
sigma = 1  # Sigma for Gaussian kernel

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    max_densities = df.groupby('step').apply(calculate_max_density, X=X, Y=Y, sigma=sigma)
    mean_max_density = max_densities.mean()
    results.append({
        'bike_pct': bike_pct,
        'e_bike_pct': e_bike_pct,
        'speedpedelec_pct': speedpedelec_pct,
        'mean_max_density': mean_max_density
    })

# Create a DataFrame from the results
final_df = pd.DataFrame(results)

# Calculate the global maximum mean maximum density
global_max_density = final_df['mean_max_density'].max()

# Normalize mean_max_density to a scale of 100
final_df['normalized_mean_max_density'] = (final_df['mean_max_density'] / global_max_density) * 100

# Prepare data for plotting
bike_group = final_df.groupby('bike_pct')['normalized_mean_max_density'].mean().reset_index()
ebike_group = final_df.groupby('e_bike_pct')['normalized_mean_max_density'].mean().reset_index()
speedpedelec_group = final_df.groupby('speedpedelec_pct')['normalized_mean_max_density'].mean().reset_index()

# Collect y_data for setting global_min and global_max
y_data = []
vehicle_info = [
    ('bike', bike_group['normalized_mean_max_density'].tolist()),
    ('ebike', ebike_group['normalized_mean_max_density'].tolist()),
    ('speed_pedelec', speedpedelec_group['normalized_mean_max_density'].tolist())
]

for _, densities in vehicle_info:
    y_data.extend(densities)

# Set global_min and global_max for y-axis
global_min = 0
global_max = 100  # Since data is normalized to 100

color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Create and display the plots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot for bike
ax = axes[0]
x = bike_group['bike_pct']
y = bike_group['normalized_mean_max_density']
ax.scatter(x, y, color=color_map['bike'], label='Data Points')
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Bike (%)')
ax.set_ylabel('Normalized Mean Maximum Density (%)')
ax.set_title('Trendline of Normalized Mean Maximum Density for Bike over all Scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for ebike
ax = axes[1]
x = ebike_group['e_bike_pct']
y = ebike_group['normalized_mean_max_density']
ax.scatter(x, y, color=color_map['ebike'], label='Data Points')
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('E-bike (%)')
ax.set_ylabel('Normalized Mean Maximum Density (%)')
ax.set_title('Trendline of Normalized Mean Maximum Density for E-bike over all Scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for speed_pedelec
ax = axes[2]
x = speedpedelec_group['speedpedelec_pct']
y = speedpedelec_group['normalized_mean_max_density']
ax.scatter(x, y, color=color_map['speed_pedelec'], label='Data Points')
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Speed Pedelec (%)')
ax.set_ylabel('Normalized Mean Maximum Density (%)')
ax.set_title('Trendline of Normalized Mean Maximum Density for Speed Pedelec over all Scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


#%% CREATE THREE OVERVIEW PLOTS OF BLIND SPOT OCCURENCE

def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

def calculate_blind_spot(group):
    x = np.stack([group["position_x"].values, group["position_y"].values], axis=1)
    
    # Calculate relative positions and angles (angle in polar coordinates)
    diffs = x[:, None, :] - x[None, :, :]
    radius = np.linalg.norm(diffs, axis=2)
    angle = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])

    # Determine the blind spot angles
    blind_spot_angles = ((angle < 5 * np.pi / 6) & (angle > np.pi / 3))   #| ((angle > 7 * np.pi / 6) & (angle < 5 * np.pi / 3))

    dead_angle = blind_spot_angles
    dead_dist = (radius < 3) & (radius != 0)

    group["dead"] = (dead_angle * dead_dist).any(axis=1)

    return group

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    
    # Calculate blind spots
    new_df = df.groupby("step").apply(calculate_blind_spot)
    grouped = new_df.groupby("id")["dead"].sum()
    
    # Merge with vehicle information
    q = df[["id", "vehicle"]].drop_duplicates(ignore_index=True)
    blind_spot_counts = pd.merge(grouped, q, on="id").groupby("vehicle").sum().reset_index()
    
    # Add scenario percentages
    blind_spot_counts['bike_pct'] = bike_pct
    blind_spot_counts['e_bike_pct'] = e_bike_pct
    blind_spot_counts['speedpedelec_pct'] = speedpedelec_pct
    
    results.append(blind_spot_counts)

final_df = pd.concat(results, ignore_index=True)

final_df['scenario'] = final_df['bike_pct'].astype(str) + '_' + final_df['e_bike_pct'].astype(str) + '_' + final_df['speedpedelec_pct'].astype(str)

pivot_df = final_df.pivot_table(index='scenario', columns='vehicle', values='dead')

# Correctly extract percentages from the scenario string
pivot_df['bike_pct'] = pivot_df.index.str.split('_').str[0].astype(int)
pivot_df['e_bike_pct'] = pivot_df.index.str.split('_').str[1].astype(int)
pivot_df['speedpedelec_pct'] = pivot_df.index.str.split('_').str[2].astype(int)

# Define vehicle types and their corresponding percentage columns
vehicle_info = [
    ('bike', 'bike_pct'),
    ('ebike', 'e_bike_pct'),
    ('speed_pedelec', 'speedpedelec_pct')
]

# Define color mapping
color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Create a figure with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for i, (vehicle, pct_col) in enumerate(vehicle_info):
    ax = axes[i]
    # Filter data for the current vehicle
    data = pivot_df[[vehicle, pct_col]].dropna()
    # Sort by the percentage column
    data = data.sort_values(by=pct_col)
    # Plot with specified color
    ax.plot(data[pct_col], data[vehicle], marker='o', color=color_map[vehicle])
    ax.set_xlabel(f'{vehicle} (%)')
    ax.set_ylabel('Blind Spot Count')
    ax.set_title(f'Blind Spot Count of {vehicle} over all scenarios')
    ax.grid(True)

plt.tight_layout()
plt.show()



# %% CREAT THREE PLOTS WITH TRENDLINE FOR BLINDSPOT OCCURENCE

def parse_scenario(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match expected format")
    bike_pct = int(parts[0])
    e_bike_pct = int(parts[1])
    speedpedelec_pct = int(parts[2])
    return bike_pct, e_bike_pct, speedpedelec_pct

def calculate_blind_spot(group):
    x = np.stack([group["position_x"].values, group["position_y"].values], axis=1)
    
    # Calculate relative positions and angles (angle in polar coordinates)
    diffs = x[:, None, :] - x[None, :, :]
    radius = np.linalg.norm(diffs, axis=2)
    angle = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])

    # Determine the blind spot angles
    blind_spot_angles = ((angle < 5 * np.pi / 6) & (angle > np.pi / 3))   #| ((angle > 7 * np.pi / 6) & (angle < 5 * np.pi / 3))

    dead_angle = blind_spot_angles
    dead_dist = (radius < 3) & (radius != 0)

    group["dead"] = (dead_angle * dead_dist).any(axis=1)

    return group

csv_files = glob.glob('logs/*.csv')
results = []

for file in csv_files:
    bike_pct, e_bike_pct, speedpedelec_pct = parse_scenario(file)
    df = pd.read_csv(file)
    
    # Calculate blind spots
    new_df = df.groupby("step").apply(calculate_blind_spot)
    grouped = new_df.groupby("id")["dead"].sum()
    
    # Merge with vehicle information
    q = df[["id", "vehicle"]].drop_duplicates(ignore_index=True)
    blind_spot_counts = pd.merge(grouped, q, on="id").groupby("vehicle").sum().reset_index()
    
    # Add scenario percentages
    blind_spot_counts['bike_pct'] = bike_pct
    blind_spot_counts['e_bike_pct'] = e_bike_pct
    blind_spot_counts['speedpedelec_pct'] = speedpedelec_pct
    
    results.append(blind_spot_counts)

final_df = pd.concat(results, ignore_index=True)

# For bike plot: group by bike_pct
bike_group = final_df[final_df['vehicle'] == 'bike'].groupby('bike_pct')['dead'].mean().reset_index()

# For ebike plot: group by e_bike_pct
ebike_group = final_df[final_df['vehicle'] == 'ebike'].groupby('e_bike_pct')['dead'].mean().reset_index()

# For speed_pedelec plot: group by speedpedelec_pct
speedpedelec_group = final_df[final_df['vehicle'] == 'speed_pedelec'].groupby('speedpedelec_pct')['dead'].mean().reset_index()

y_data = []
vehicle_info = [
    ('bike', bike_group['dead'].tolist()),
    ('ebike', ebike_group['dead'].tolist()),
    ('speed_pedelec', speedpedelec_group['dead'].tolist())
]

for _, counts in vehicle_info:
    y_data.extend(counts)

global_min = np.nanmin(y_data) - 0.5
global_max = np.nanmax(y_data) + 0.5

color_map = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot for bike
ax = axes[0]
x = bike_group['bike_pct']
y = bike_group['dead']
ax.scatter(x, y, color=color_map['bike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Bike (%)')
ax.set_ylabel('Blind Spot Count')
ax.set_title('Trendline Blind Spot Count of Bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for ebike
ax = axes[1]
x = ebike_group['e_bike_pct']
y = ebike_group['dead']
ax.scatter(x, y, color=color_map['ebike'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('E-bike (%)')
ax.set_ylabel('Blind Spot Count')
ax.set_title('Trendline Blind Spot Count of E-bike over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

# Plot for speed_pedelec
ax = axes[2]
x = speedpedelec_group['speedpedelec_pct']
y = speedpedelec_group['dead']
ax.scatter(x, y, color=color_map['speed_pedelec'], label='Data Points')
# Trendline
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
x_trend = np.linspace(x.min(), x.max(), 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='black', linestyle='-', linewidth=2, label='Trendline')
ax.set_xlabel('Speed Pedelec (%)')
ax.set_ylabel('Blind Spot Count')
ax.set_title('Trendline Blind Spot Count of Speed Pedelec over all scenarios')
ax.set_ylim(global_min, global_max)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


