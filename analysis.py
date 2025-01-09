# %% LOAD EVERYTHING & INITIALISE
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv("old_logs/scaleup.csv")

# Adapt velocity
df["velocity"] = (df["velocity_x"]**2 + df["velocity_y"]**2)**0.5
df["acceleration"] = (df["acceleration_x"]**2 + df["acceleration_y"]**2)**0.5

# Get the data grouped by vehicle
groups = df.groupby("vehicle")

# %% SPEED: DISTPLOT with velocity NORM density per vehicle TYPE

# Extract the acceleration data for each vehicle type
hist_data = []
group_labels = []
for name, group in groups:
    hist_data.append(group["velocity"])
    group_labels.append(name)

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(
    title="Distribution of Velocity for Different Vehicle Types 0/100",
    xaxis_title="Velocity (m/s)",
    yaxis_title="Density"
)
fig.show()


# %% DEVIATION: DISTPLOT with MEAN per vehicle TYPE
# Calculate deviation from preferred y-position mean per vehicle type

preferred_y = -1.5

# Extract the acceleration data for each vehicle type
hist_data = []
group_labels = []
for name, group in groups:
    hist_data.append(group["position_y"])
    group_labels.append(name)

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(
    title="Distribution of Perpendicular Deviation for Different Vehicle Types 50/50",
    xaxis_title="Y-Position Deviation (m)",
    yaxis_title="Density"
)
fig.add_vline(x=preferred_y, line_width=2, line_dash="solid", line_color="red")
fig.show()

 # %% BRAKING: DISTPLOT of braking per vehicle TYPE
# Distribution of braking events' velocity by vehicle type
# Get the brake data grouped by vehicle
brake = df[df["acceleration_x"] < 0.0]
brake_groups = brake.groupby("vehicle")

heavy_brake = df[df["acceleration_x"] < -0.2]
heavy_brake_groups = heavy_brake.groupby("vehicle")

# Extract the acceleration data for each vehicle type
hist_data = []
group_labels = []
for name, group in heavy_brake_groups:
    hist_data.append(group["velocity"])
    group_labels.append(name)

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(
    title="Distribution of Velocities for Different Vehicle Types with Heavy Braking",
    xaxis_title="Velocity (m/s)",
    yaxis_title="Density"
)
fig.show()

# %% DENSITY: VALUE MEAN MAX of density per time STEP with GAUSSIAN
# Value for mean of max density per time step

def create_grid(group):
    grid = np.zeros((1000, 24))
    pos_x = group["position_x"].tolist()
    pos_y = group["position_y"].tolist()
    
    for x, y in zip(pos_x, pos_y):
        grid[np.clip(int(x * 10), 0, 999), np.clip(int((y + 2) * 6), 0, 23)] = 1
    
    k = 5
    grid = gaussian_filter(grid, sigma=[10*k, k])
    grid = (grid * 1e3) ** 4
    
    return grid

# Function to calculate density
def calculate_density(group):
    return create_grid(group).flatten().max()

# Calculate max density per timestep
max_densities = df.groupby('step').apply(calculate_density)

# Calculate mean of max densities
mean_max_density = max_densities.mean()

print(f"Mean of maximum densities: {mean_max_density}")

# %% DEVIATION: VALUE with mean of sum (INTEGRAL)

grouped = df.groupby("id")

# check per id for which steps the sum needs to be taken
mapped = grouped["position_y"].sum() / (grouped["step"].max() - grouped["step"].min())

# aggregate over vehicle type
v = pd.DataFrame(grouped["vehicle"].first())
v["value"] = mapped.values

# take mean of integral per vehicle type
v.groupby("vehicle")["value"].mean()

# %% SPEED: VALUE mean per vehicle TYPE  

print(df.groupby("vehicle")["velocity"].mean())

#%% BRAKING: VALUE mean per vehicle TYPE

df["decelerate"] = df["acceleration_x"] < 0.0
df["braking"] = df["acceleration_x"] < -0.2

print('- Deceleration %', df.groupby("vehicle")["decelerate"].mean())
print('- Heavy braking %', df.groupby("vehicle")["braking"].mean())
print('- Heavy braking of deceleration %', df.groupby("vehicle")["braking"].mean() *100 / df.groupby("vehicle")["decelerate"].mean())

# %% BLIND SPOT: VALUE with mean per vehicle TYPE

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

new = df.groupby("step").apply(calculate_blind_spot)
grouped = new.groupby("id")["dead"].sum()

q = df[["id", "vehicle"]].drop_duplicates(ignore_index=True)
new = pd.merge(grouped, q, on="id").groupby("vehicle").sum()

print(new)

# %% TRAJECTORIES

# Get unique vehicle IDs (assuming 'id' is the column for unique vehicle IDs)
vehicle_ids = df['id'].unique()

# Create a plot
plt.figure(figsize=(20, 10))

# Plot trajectories for each unique vehicle ID
for vehicle_id in vehicle_ids:
    vehicle_data = df[df['id'] == vehicle_id]
    plt.plot(vehicle_data['position_x'], vehicle_data['position_y'], label=f'Vehicle ID {vehicle_id}')

# Add labels and title
plt.xlabel('Position X (meters)')
plt.ylabel('Position Y (meters)')
plt.title('Vehicle Trajectories by ID')

# Set axis limits to ensure the plot shows up to 200 meters
plt.xlim(0, 200)  # Set x-axis limits from 0 to 200 meters
plt.ylim(-2, 2)  # Set y-axis limits from 0 to 200 meters

# Add a legend and grid
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


# %% HISTORGRAM SPEED

# Calculate speed for each vehicle
df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

# Define colors for each vehicle type
colors = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Create a figure with 3 subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Calculate the maximum y-value across all histograms
max_y = 0
for vehicle_type in colors.keys():
    vehicle_data = df[df['vehicle'] == vehicle_type]
    counts, _ = np.histogram(vehicle_data['speed'], bins=20, density=True)
    max_y = max(max_y, np.max(counts))

# Plot a histogram for each vehicle type in its own subplot
for i, (vehicle_type, color) in enumerate(colors.items()):
    ax = axes[i]
    vehicle_data = df[df['vehicle'] == vehicle_type]
    
    # Normalize the histogram by dividing by the number of instances
    ax.hist(vehicle_data['speed'], bins=20, color=color, alpha=0.7, density=True)
    
    # Add labels and title
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Normalized Frequency')
    ax.set_title(f'Velocity Distribution: {vehicle_type.capitalize()}')
    ax.grid(True)
    
    # Set the same y-axis limits for all subplots
    ax.set_ylim(0, max_y * 1.1)  # Add 10% padding to the max y-value

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# %% TREND speed line normalised

# Calculate speed for each vehicle
df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

# Define colors for each vehicle type
colors = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Create a single plot
plt.figure(figsize=(10, 6))

# Plot a KDE for each vehicle type
for vehicle_type, color in colors.items():
    vehicle_data = df[df['vehicle'] == vehicle_type]
    sns.kdeplot(vehicle_data['speed'], color=color, label=vehicle_type.capitalize(), linewidth=2)

# Add labels and title
plt.xlabel('Velocity (m/s)')
plt.ylabel('Density')
plt.title('Velocity Distribution by Vehicle Type')
plt.legend()

# Add a grid
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# %% MEAN SPEED PER VEHICLE TYPE


# Calculate speed for each vehicle
df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

# Define colors for each vehicle type
colors = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Calculate mean speed for each vehicle type
mean_speeds = df.groupby('vehicle')['speed'].mean()

# Create a single histogram (bar plot) for mean speeds
plt.figure(figsize=(8, 6))

# Plot bars for each vehicle type
for vehicle_type, color in colors.items():
    plt.bar(vehicle_type, mean_speeds[vehicle_type], color=color, label=vehicle_type.capitalize())

# Add labels and title
plt.xlabel('Vehicle Type')
plt.ylabel('Mean Speed (m/s)')
plt.title('Mean Speed by Vehicle Type')
plt.legend()

# Add a grid
plt.grid(True, axis='y')

# Show the plot
plt.tight_layout()
plt.show()
# %% MEAN DEVIATION PER VEHICLE TYPE
# Define colors for each vehicle type
colors = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Calculate mean y-deviation for each vehicle type
mean_y_deviation = df.groupby('vehicle')['position_y'].mean()

# Create a single histogram (bar plot) for mean y-deviation
plt.figure(figsize=(8, 6))

# Plot bars for each vehicle type
for vehicle_type, color in colors.items():
    plt.bar(vehicle_type, mean_y_deviation[vehicle_type], color=color, label=vehicle_type.capitalize())

# Add labels and title
plt.xlabel('Vehicle Type')
plt.ylabel('Mean Y-Deviation (meters)')
plt.title('Mean Y-Deviation by Vehicle Type')
plt.legend()

# Add a grid
plt.grid(True, axis='y')

# Show the plot
plt.tight_layout()
plt.show()

#%% DEVIATION DISTRIBUTION

# Define colors for each vehicle type
colors = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}


# Ensure 'position_y' has no NaN values
df = df.dropna(subset=['position_y'])

# Determine overall x-axis limits
overall_min = df['position_y'].min()
overall_max = df['position_y'].max()

# Create a figure with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot for bike
bike_data = df[df['vehicle'] == 'bike']['position_y']
sns.kdeplot(bike_data, ax=axes[0], color=colors['bike'], fill=True)
axes[0].set_xlabel('Y-Deviation (meters)')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution of Y-Deviation for Bike')
axes[0].set_xlim(overall_min, overall_max)
axes[0].grid(True)

# Plot for ebike
ebike_data = df[df['vehicle'] == 'ebike']['position_y']
sns.kdeplot(ebike_data, ax=axes[1], color=colors['ebike'], fill=True)
axes[1].set_xlabel('Y-Deviation (meters)')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Y-Deviation for Ebike')
axes[1].set_xlim(overall_min, overall_max)
axes[1].grid(True)

# Plot for speed_pedelec
speedpedelec_data = df[df['vehicle'] == 'speed_pedelec']['position_y']
sns.kdeplot(speedpedelec_data, ax=axes[2], color=colors['speed_pedelec'], fill=True)
axes[2].set_xlabel('Y-Deviation (meters)')
axes[2].set_ylabel('Density')
axes[2].set_title('Distribution of Y-Deviation for Speed Pedelec')
axes[2].set_xlim(overall_min, overall_max)
axes[2].grid(True)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# %% MEAN ACCELERATION AND HEAVY BRAKING
# Calculate acceleration magnitude
df['acceleration'] = df['acceleration_x']

# Define colors for each vehicle type
colors = {
    'bike': 'blue',
    'ebike': 'pink',
    'speed_pedelec': 'green'
}

# Create a figure with 3 subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Calculate the maximum y-value across all histograms for consistent y-axis limits
max_y = 0
for vehicle_type in colors.keys():
    vehicle_data = df[df['vehicle'] == vehicle_type]
    counts, _ = np.histogram(vehicle_data['acceleration'], bins=20, density=True)
    max_y = max(max_y, np.max(counts))

# Plot a histogram for each vehicle type in its own subplot
for i, (vehicle_type, color) in enumerate(colors.items()):
    ax = axes[i]
    vehicle_data = df[df['vehicle'] == vehicle_type]
    
    # Plot the histogram (without KDE)
    ax.hist(vehicle_data['acceleration'], bins=20, color=color, alpha=0.7, density=True, label=vehicle_type.capitalize())
    
    # Add a vertical line to indicate the heavy braking threshold
    ax.axvline(x=-2, color='red', linestyle='--', label='Heavy Braking Threshold (-2 m/s²)')
    
    # Add labels and title
    ax.set_xlabel('Acceleration (m/s²)')
    ax.set_ylabel('Density')
    ax.set_title(f'Acceleration Distribution: {vehicle_type.capitalize()}')
    ax.legend()
    ax.grid(True)
    
    # Set the same y-axis limits for all subplots
    ax.set_ylim(0, max_y * 1.1)  # Add 10% padding to the max y-value

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
# %% DENSITY GRAPH & VALUE FOR BASIC INSIGHTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Function to create a grid and calculate density
def create_grid(group):
    grid = np.zeros((1000, 24))
    pos_x = group["position_x"].tolist()
    pos_y = group["position_y"].tolist()
    
    for x, y in zip(pos_x, pos_y):
        grid[np.clip(int(x * 10), 0, 999), np.clip(int((y + 2) * 6), 0, 23)] = 1
    
    k = 5
    grid = gaussian_filter(grid, sigma=[10 * k, k])
    grid = (grid * 1e3) ** 4
    
    return grid

# Function to calculate maximum density for a group
def calculate_density(group):
    return create_grid(group).flatten().max()
# Calculate max density per timestep
max_densities = df.groupby('step').apply(calculate_density)

# Calculate mean of max densities
mean_max_density = max_densities.mean()
print(f"Mean of maximum densities: {mean_max_density}")

# Plot the trend of max densities over time
plt.figure(figsize=(12, 6))
plt.plot(max_densities.index, max_densities.values, color='blue', label='Max Density per Time Step')
plt.axhline(mean_max_density, color='red', linestyle='--', label=f'Mean Max Density: {mean_max_density:.2f}')

# Add labels, title, and legend
plt.xlabel('Time Step')
plt.ylabel('Maximum Density')
plt.title('Trend of Maximum Density Over Time for a Single Scenario')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# %% BLIND SPOT BASIC INSIGHTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate blind spot occurrences
def calculate_blind_spot(group):
    x = np.stack([group["position_x"].values, group["position_y"].values], axis=1)
    
    # Calculate relative positions and angles (angle in polar coordinates)
    diffs = x[:, None, :] - x[None, :, :]
    radius = np.linalg.norm(diffs, axis=2)
    angle = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])

    # Determine the blind spot angles
    blind_spot_angles = ((angle < 5 * np.pi / 6) & (angle > np.pi / 3))  # Define blind spot angles
    dead_angle = blind_spot_angles
    dead_dist = (radius < 3) & (radius != 0)  # Define blind spot distance threshold

    group["dead"] = (dead_angle * dead_dist).any(axis=1)  # Mark agents in blind spots
    return group

# Apply calculate_blind_spot to each time step
new = df.groupby("step").apply(calculate_blind_spot)

# Calculate the total number of blind spot occurrences per time step
blind_spot_counts = new.groupby("step")["dead"].sum()

# Calculate the mean number of blind spot occurrences
mean_blind_spot = blind_spot_counts.mean()
print(f"Mean number of blind spot occurrences: {mean_blind_spot}")

# Plot the trend of blind spot occurrences over time
plt.figure(figsize=(12, 6))
plt.plot(blind_spot_counts.index, blind_spot_counts.values, color='blue', label='Blind Spot Occurrences per Time Step')
plt.axhline(mean_blind_spot, color='red', linestyle='--', label=f'Mean Blind Spot Occurrences: {mean_blind_spot:.2f}')

# Add labels, title, and legend
plt.xlabel('Time Step')
plt.ylabel('Number of Blind Spot Occurrences')
plt.title('Trend of Blind Spot Occurrences Over Time for a Single Scenario')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# %%
