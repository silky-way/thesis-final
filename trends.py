# %% main run
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import numpy as np

# %% Plot speed trendlines per vehicle type over all simulations
def plot_speed_trendlines(csv_folder_path):
    # Find all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))
    
    # Dictionary to store aggregated data for each vehicle type
    aggregated_data = {}
    
    # Process each CSV file
    for file_path in csv_files:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Calculate total velocity
        df['total_velocity'] = ((df['velocity_x']**2 + df['velocity_y']**2)**0.5)
        
        # Group by vehicle type and step
        grouped_data = df.groupby(['vehicle', 'step'])['total_velocity'].mean().reset_index()
        
        # Store grouped data
        for vehicle_type in grouped_data['vehicle'].unique():
            if vehicle_type not in aggregated_data:
                aggregated_data[vehicle_type] = []
            
            vehicle_group = grouped_data[grouped_data['vehicle'] == vehicle_type]
            aggregated_data[vehicle_type].append(vehicle_group)
    
    # Create subplots for each vehicle type
    fig, axes = plt.subplots(len(aggregated_data), 1, figsize=(12, 4*len(aggregated_data)), sharex=True)
    
    # Ensure axes is always a list, even if there's only one vehicle type
    if len(aggregated_data) == 1:
        axes = [axes]
    
    # Plot for each vehicle type
    for idx, (vehicle_type, data_list) in enumerate(aggregated_data.items()):
        # Combine all runs for this vehicle type
        combined_data = pd.concat(data_list)
        
        # Group by step and calculate mean and standard deviation
        summary_data = combined_data.groupby('step')['total_velocity'].agg(['mean', 'std']).reset_index()
        
        # Plot mean with error bands
        axes[idx].plot(summary_data['step'], summary_data['mean'], label='Mean Velocity')
        axes[idx].fill_between(summary_data['step'], 
                                summary_data['mean'] - summary_data['std'], 
                                summary_data['mean'] + summary_data['std'], 
                                alpha=0.3, label='Standard Deviation')
        
        axes[idx].set_title(f'Velocity Trendline - {vehicle_type}')
        axes[idx].set_ylabel('Average Velocity (m/s)')
        axes[idx].legend()
        axes[idx].grid(True, linestyle='--', alpha=0.7)
    
    # Set common x-label
    axes[-1].set_xlabel('Simulation Step')
    
    plt.tight_layout()
    plt.show()


    # Example usage
plot_speed_trendlines('logs')



#%% Plot vehicle distributions and their mean velocity -- SPEEDPEDELECS DO NOT SHOW
def extract_distribution(filename):
    match = re.search(r'(\d+)_(\d+)_(\d+)', filename)
    if match:
        return {
            'bike': int(match.group(1)),
            'ebike': int(match.group(2)),
            'pedelec': int(match.group(3))
        }
    return None

def analyze_velocity_distribution(csv_folder_path):
    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))
    
    # Prepare data structures
    vehicle_types = ['bike', 'ebike', 'pedelec']
    distribution_analysis = {vtype: {'percentages': [], 'mean_velocities': []} for vtype in vehicle_types}
    
    # Process each CSV file
    for file_path in csv_files:
        # Extract distribution
        distribution = extract_distribution(os.path.basename(file_path))
        if not distribution:
            continue
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Calculate total velocity
        df['total_velocity'] = ((df['velocity_x']**2 + df['velocity_y']**2)**0.5)
        
        # Analyze for each vehicle type
        for vehicle_type in vehicle_types:
            # Filter for specific vehicle type and calculate mean velocity
            vehicle_data = df[df['vehicle'] == vehicle_type]
            mean_velocity = vehicle_data['total_velocity'].mean()
            
            distribution_analysis[vehicle_type]['percentages'].append(distribution[vehicle_type])
            distribution_analysis[vehicle_type]['mean_velocities'].append(mean_velocity)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, vehicle_type in enumerate(vehicle_types):
        data = distribution_analysis[vehicle_type]
        axes[idx].scatter(data['percentages'], data['mean_velocities'], alpha=0.7)
        axes[idx].set_xlabel('Vehicle Distribution (%)')
        axes[idx].set_ylabel('Mean Velocity (m/s)')
        axes[idx].set_title(f'Velocity vs Distribution - {vehicle_type.capitalize()}')
        
        # Optional: Add trendline
        if len(data['percentages']) > 1:
            z = np.polyfit(data['percentages'], data['mean_velocities'], 1)
            p = np.poly1d(z)
            axes[idx].plot(data['percentages'], p(data['percentages']), "r--", label='Trend')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
analyze_velocity_distribution('logs')
# %% Y position trendline per vehicle type over all simulations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_y_position_trendlines(csv_folder_path):
    # Find all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))
    
    # Set up the plot
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Dictionary to store aggregated data
    aggregated_data = {}
    
    # Process each CSV file
    for file_path in csv_files:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract filename (without extension) as run identifier
        run_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Group by vehicle type and step
        grouped_data = df.groupby(['vehicle', 'step'])['position_y'].mean().reset_index()
        
        # Store grouped data
        for vehicle_type in grouped_data['vehicle'].unique():
            if vehicle_type not in aggregated_data:
                aggregated_data[vehicle_type] = []
            
            vehicle_group = grouped_data[grouped_data['vehicle'] == vehicle_type]
            vehicle_group['run'] = run_name
            aggregated_data[vehicle_type].append(vehicle_group)
    
    # Plot trendlines for each vehicle type
    for vehicle_type, data_list in aggregated_data.items():
        combined_data = pd.concat(data_list)
        
        # Calculate mean and standard deviation for the y-position
        summary_data = combined_data.groupby('step')['position_y'].agg(['mean', 'std']).reset_index()
        
        # Plot mean with error bands
        plt.plot(summary_data['step'], summary_data['mean'], label=vehicle_type)
        plt.fill_between(summary_data['step'], 
                         summary_data['mean'] - summary_data['std'], 
                         summary_data['mean'] + summary_data['std'], 
                         alpha=0.3)
    
    plt.title('Y Position Trendlines Across Multiple Runs')
    plt.xlabel('Simulation Step')
    plt.ylabel('Average Y Position')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
plot_y_position_trendlines('logs')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

def load_csv_data(csv_folder_path):
    """Load and combine CSV files from a folder"""
    csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))
    dataframes = [pd.read_csv(f) for f in csv_files]
    
    # Calculate total velocity
    for df in dataframes:
        df['total_velocity'] = ((df['velocity_x']**2 + df['velocity_y']**2)**0.5)
    
    return dataframes

def velocity_distribution_analysis(dataframes):
    """Create comprehensive velocity distribution visualization"""
    plt.figure(figsize=(15, 5))
    
    # Violin plot
    plt.subplot(131)
    vehicle_types = ['bike', 'ebike', 'pedelec']
    vehicle_velocities = {vehicle: [] for vehicle in vehicle_types}
    
    for df in dataframes:
        for vehicle in vehicle_types:
            # Only extend if the vehicle type exists in this dataframe
            vehicle_data = df[df['vehicle'] == vehicle]['total_velocity']
            if not vehicle_data.empty:
                vehicle_velocities[vehicle].extend(vehicle_data)
    
    # Filter out empty lists
    plot_data = [vels for vels in vehicle_velocities.values() if len(vels) > 0]
    plot_labels = [v for v, vels in vehicle_velocities.items() if len(vels) > 0]
    
    if plot_data:
        plt.violinplot(plot_data)
        plt.title('Velocity Distribution')
        plt.xticks(range(1, len(plot_labels) + 1), [v.capitalize() for v in plot_labels])
        plt.ylabel('Velocity (m/s)')
    
    # Cumulative Distribution Function
    plt.subplot(132)
    for vehicle, color in zip(plot_labels, ['blue', 'green', 'red']):
        vehicle_data = np.concatenate(
            [df[df['vehicle'] == vehicle]['total_velocity'] for df in dataframes 
             if not df[df['vehicle'] == vehicle].empty]
        )
        if len(vehicle_data) > 0:
            vehicle_data.sort()
            cdf = np.arange(1, len(vehicle_data) + 1) / len(vehicle_data)
            plt.plot(vehicle_data, cdf, label=vehicle.capitalize(), color=color)
    
    plt.title('Cumulative Velocity Distribution')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    
    # Box plot of maximum interaction forces
    plt.subplot(133)
    interaction_forces = {vehicle: [] for vehicle in vehicle_types}
    
    for df in dataframes:
        for vehicle in vehicle_types:
            vehicle_forces = df[df['vehicle'] == vehicle]['max_interaction_force']
            if not vehicle_forces.empty:
                interaction_forces[vehicle].extend(vehicle_forces)
    
    # Filter out empty lists
    plot_forces = [forces for forces in interaction_forces.values() if len(forces) > 0]
    plot_force_labels = [v for v, forces in interaction_forces.items() if len(forces) > 0]
    
    if plot_forces:
        plt.boxplot(plot_forces)
        plt.title('Maximum Interaction Forces')
        plt.xticks(range(1, len(plot_force_labels) + 1), 
                   [v.capitalize() for v in plot_force_labels])
        plt.ylabel('Force Magnitude')
    
    plt.tight_layout()
    plt.show()

def correlation_heatmap(dataframes):
    """Create correlation heatmap of key variables"""
    # Combine all dataframes
    combined_df = pd.concat(dataframes)
    
    # Select numeric columns
    numeric_cols = [
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y', 
        'rolling_velocity_x'
    ]
    
    # Compute correlation
    correlation_matrix = combined_df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Variable Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def trajectory_analysis(dataframes):
    """Analyze and visualize vehicle trajectories"""
    plt.figure(figsize=(15, 5))
    
    vehicle_types = ['bike', 'ebike', 'pedelec']
    
    # 2D trajectory plot
    plt.subplot(121)
    for df in dataframes:
        for vehicle in vehicle_types:
            vehicle_data = df[df['vehicle'] == vehicle]
            if not vehicle_data.empty:
                plt.scatter(vehicle_data['position_x'], vehicle_data['position_y'], 
                            label=vehicle, alpha=0.5)
    
    plt.title('Vehicle Trajectory Scatter')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    
    # Position variation boxplot
    plt.subplot(122)
    position_variation = {vehicle: [] for vehicle in vehicle_types}
    for df in dataframes:
        for vehicle in vehicle_types:
            vehicle_data = df[df['vehicle'] == vehicle]
            if not vehicle_data.empty:
                position_variation[vehicle].append([
                    np.std(vehicle_data['position_x']),
                    np.std(vehicle_data['position_y'])
                ])
    
    # Filter out empty lists
    plot_variation = [vars for vars in position_variation.values() if len(vars) > 0]
    plot_var_labels = [v for v, vars in position_variation.items() if len(vars) > 0]
    
    if plot_variation:
        plt.boxplot(plot_variation)
        plt.title('Position Variation')
        plt.xticks(range(1, len(plot_var_labels) + 1), 
                   [v.capitalize() for v in plot_var_labels])
        plt.ylabel('Position Standard Deviation')
    
    plt.tight_layout()
    plt.show()

# Example usage
csv_folder_path = 'logs'
dataframes = load_csv_data(csv_folder_path)

# Run different analyses
velocity_distribution_analysis(dataframes)
correlation_heatmap(dataframes)
trajectory_analysis(dataframes)
# %%
