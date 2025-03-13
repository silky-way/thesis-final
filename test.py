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
df = pd.read_csv("bigfat.csv")

# print(df)


#%% TTC 

def compute_ttc(df):
    """
    Compute Time to Collision (TTC) for each vehicle pair.
    """
    ttc_data = []

    # Group by step (to compare vehicles at the same moment in time)
    for step, step_df in df.groupby('step'):
        vehicles = step_df.to_dict('records')

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):  # Compare each vehicle with another
                v1 = vehicles[i]
                v2 = vehicles[j]

                # Compute relative velocity and position difference
                delta_x = v2["position_x"] - v1["position_x"]
                delta_v = v2["velocity_x"] - v1["velocity_x"]

                # Compute TTC if there is a risk of collision
                if delta_v > 0:  # Only consider vehicles that are approaching each other
                    ttc = abs(delta_x / delta_v)
                else:
                    ttc = np.inf  # No collision risk

                ttc_data.append({"step": step, "vehicle_1": v1["id"], "vehicle_2": v2["id"], "TTC": ttc})

    return pd.DataFrame(ttc_data)

# Compute TTC
ttc_df = compute_ttc(df)

# Pivot for heatmap
# Handle duplicates by averaging TTC values
pivot_ttc = ttc_df[ttc_df["TTC"] < 2].groupby(["vehicle_1", "vehicle_2"])["TTC"].count().unstack()

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_ttc, cmap="coolwarm", linewidths=0.5)
plt.title("Time to Collision (TTC) Heatmap")
plt.xlabel("Vehicle 2")
plt.ylabel("Vehicle 1")
plt.show()


# %% space-timeplot 1

# Ensure correct data types
df["step"] = df["step"].astype(int)  # Time variable
df["position_x"] = df["position_x"].astype(float)  # Space variable

# Filter for the specific vehicle ID
vehicle_id = 68  # Change this to any ID you want to plot
vehicle_df = df[df["id"] == vehicle_id]

# Check if the ID exists in the data
if vehicle_df.empty:
    print(f"Vehicle ID {vehicle_id} not found in the dataset.")
else:
    # Sort values for proper plotting
    vehicle_df = vehicle_df.sort_values(by=["step"])

    # Get vehicle type and assign a color
    vehicle_type = vehicle_df["vehicle"].iloc[0]  # Get vehicle type
    vehicle_colors = {"bike": "blue", "ebike": "green", "speed_pedelec": "red"}
    color = vehicle_colors.get(vehicle_type, "black")  # Default color if type not in dict

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create Space-Time Plot for the specific vehicle
    plt.figure(figsize=(12, 8))
    plt.plot(vehicle_df["step"], vehicle_df["position_x"], label=f"ID {vehicle_id} ({vehicle_type})", color=color, alpha=0.7)

    # Labels and Title
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Position (x)", fontsize=12)
    plt.title(f"Space-Time Diagram for Vehicle ID {vehicle_id}", fontsize=14)
    plt.legend(title="Vehicle Type", fontsize="medium")
    plt.show()
# %% space-time plot all
# Ensure correct data types
df["step"] = df["step"].astype(int)  # Time variable
df["position_x"] = df["position_x"].astype(float)  # Space variable

# Sort values for proper plotting
df = df.sort_values(by=["id", "step"])  # Ensure correct order per vehicle

# Set seaborn style
sns.set_style("whitegrid")

# Define a color palette for vehicle types
vehicle_colors = {"bike": "blue", "ebike": "green", "speed_pedelec": "red"}

# Create Space-Time Plot
plt.figure(figsize=(12, 8))

# Loop over each unique vehicle ID and plot its trajectory
for vehicle_id in df["id"].unique():
    vehicle_df = df[df["id"] == vehicle_id]
    vehicle_type = vehicle_df["vehicle"].iloc[0]  # Get vehicle type
    color = vehicle_colors.get(vehicle_type, "black")  # Default color if type not in dict
    plt.plot(vehicle_df["step"], vehicle_df["position_x"], label=vehicle_type if vehicle_id == df["id"].unique()[0] else "", color=color, alpha=0.7)

# Labels and Title
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Position (x)", fontsize=12)
plt.title("Space-Time Diagram by Vehicle ID", fontsize=14)
plt.legend(title="Vehicle Type", fontsize="medium")
plt.show()

# %% passing maneuvre count

# Ensure correct data types
df["step"] = df["step"].astype(int)
df["position_x"] = df["position_x"].astype(float)
df["position_y"] = df["position_y"].astype(float)
df["velocity_x"] = df["velocity_x"].astype(float)

# FILTER: Ignore data where position_x <= 10 (initialization period)
df = df[df["position_x"] > 10]

# Define passing maneuver detection threshold
x_threshold = 2.0  # Two vehicles are considered in the same x region if within this range

# Sort values for proper processing
df = df.sort_values(by=["step", "position_x"])

# Check for negative speeds
# negative_speeds = df[df["velocity_x"] < 0]
# if not negative_speeds.empty:
#     print("⚠️ Warning: Negative speeds detected!")
#     print(negative_speeds["vehicle"].value_counts())  # Count negative speeds per vehicle type
#     print("Some rows with negative speeds:")
#     print(negative_speeds.head(10))  # Print a sample for debugging

# Remove negative speeds
# df = df[df["velocity_x"] >= 0]

# Find passing maneuvers
passing_maneuvers = []
processed_pairs = set()

for step in df["step"].unique():
    step_df = df[df["step"] == step]  # Vehicles at the same time step
    
    # Iterate over all vehicle pairs
    for i, veh1 in step_df.iterrows():
        for j, veh2 in step_df.iterrows():
            if i >= j:  # Avoid duplicate comparisons
                continue
            
            # Check if vehicles are close in x but one is higher in y (passing maneuver)
            if abs(veh1["position_x"] - veh2["position_x"]) <= x_threshold:
                passing_vehicle = veh1 if veh1["position_y"] > veh2["position_y"] else veh2
                passed_vehicle = veh2 if veh1["position_y"] > veh2["position_y"] else veh1
                
                # Ensure we don't double count the same pair over multiple steps
                pair_id = tuple(sorted([passing_vehicle["id"], passed_vehicle["id"]]))
                if pair_id not in processed_pairs:
                    passing_maneuvers.append(passing_vehicle)
                    processed_pairs.add(pair_id)

# Convert passing maneuvers into DataFrame
passing_df = pd.DataFrame(passing_maneuvers)

# Count passing maneuvers per vehicle type
passing_counts = passing_df["vehicle"].value_counts()

# Print the passing maneuver count per vehicle type
print("\n🚴 Passing Maneuvers per Vehicle Type:")
print(passing_counts)

# Set seaborn style
sns.set_style("whitegrid")

# Define color map
color_map = {"bike": "blue", "ebike": "green", "speed_pedelec": "red"}

# Plot speed distribution per vehicle type
plt.figure(figsize=(12, 6))

for vehicle_type, color in color_map.items():
    sns.kdeplot(passing_df[passing_df["vehicle"] == vehicle_type]["velocity_x"], 
                label=vehicle_type, color=color, fill=True, alpha=0.5)

# Labels and Title
plt.xlabel("Speed (m/s)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Speed Distribution During Passing Maneuvers", fontsize=14)
plt.legend(title="Vehicle Type", fontsize="medium")
plt.grid(True)

plt.show()

# %% # Ensure correct data types
df["step"] = df["step"].astype(int)
df["position_x"] = df["position_x"].astype(float)
df["position_y"] = df["position_y"].astype(float)
df["acceleration_x"] = df["acceleration_x"].astype(float)

# FILTER: Ignore data where position_x <= 10 (initialization period)
df = df[df["position_x"] > 10]

# Define passing maneuver detection threshold
x_threshold = 2.0  # Vehicles are considered in the same x region if within this range

# Sort values for proper processing
df = df.sort_values(by=["step", "position_x"])

# Find passing maneuvers
passing_maneuvers = []
processed_pairs = set()

for step in df["step"].unique():
    step_df = df[df["step"] == step]  # Vehicles at the same time step
    
    # Iterate over all vehicle pairs
    for i, veh1 in step_df.iterrows():
        for j, veh2 in step_df.iterrows():
            if i >= j:  # Avoid duplicate comparisons
                continue
            
            # Check if vehicles are close in x but one is higher in y (passing maneuver)
            if abs(veh1["position_x"] - veh2["position_x"]) <= x_threshold:
                passing_vehicle = veh1 if veh1["position_y"] > veh2["position_y"] else veh2
                passed_vehicle = veh2 if veh1["position_y"] > veh2["position_y"] else veh1
                
                # Ensure we don't double count the same pair over multiple steps
                pair_id = tuple(sorted([passing_vehicle["id"], passed_vehicle["id"]]))
                if pair_id not in processed_pairs:
                    passing_maneuvers.append(passing_vehicle)
                    processed_pairs.add(pair_id)

# Convert passing maneuvers into DataFrame
passing_df = pd.DataFrame(passing_maneuvers)

# Count passing maneuvers per vehicle type
passing_counts = passing_df["vehicle"].value_counts()

# Print the passing maneuver count per vehicle type
print("\n🚴 Passing Maneuvers per Vehicle Type:")
print(passing_counts)

# Set seaborn style
sns.set_style("whitegrid")

# Define color map
color_map = {"bike": "blue", "ebike": "green", "speed_pedelec": "red"}

# Plot acceleration distribution per vehicle type
plt.figure(figsize=(12, 6))

for vehicle_type, color in color_map.items():
    sns.kdeplot(passing_df[passing_df["vehicle"] == vehicle_type]["acceleration_x"], 
                label=vehicle_type, color=color, fill=True, alpha=0.5)

# Labels and Title
plt.xlabel("Acceleration (m/s²)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Acceleration Distribution During Passing Maneuvers (x > 10)", fontsize=14)
plt.legend(title="Vehicle Type", fontsize="medium")
plt.grid(True)

plt.show()


# %%
