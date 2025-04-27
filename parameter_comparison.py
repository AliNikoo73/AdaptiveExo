#!/usr/bin/env python3
"""
Compares original exoskeleton parameters for subject 10 with the personalized 
parameters after adaptation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set_context("talk")

# Create output directory
os.makedirs('results/comparison', exist_ok=True)

# Load the simulation results
results_df = pd.read_csv('results/simulation_results.csv')

# Get subject 10 data
subject_10_df = results_df[results_df['subject_id'] == 10]

# Find the best (lowest metabolic cost) parameter set
best_params = subject_10_df.loc[subject_10_df['metabolic_cost'].idxmin()]

# Find the worst (highest metabolic cost) parameter set
worst_params = subject_10_df.loc[subject_10_df['metabolic_cost'].idxmax()]

# Parameter columns to compare
param_cols = ['onset_percent', 'offset_percent', 'peak_torque', 'rise_time', 
              'fall_time', 'stiffness', 'damping', 'knee_assist_weight', 
              'hip_assist_weight', 'ankle_assist_weight']

# Extract parameter values
best_values = best_params[param_cols].values
worst_values = worst_params[param_cols].values

# 1. Bar chart to compare metabolic cost and assistance efficiency
metrics = ['metabolic_cost', 'assistance_efficiency']
best_metrics = best_params[metrics].values
worst_metrics = worst_params[metrics].values

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, worst_metrics, width, label='Original Parameters')
ax.bar(x + width/2, best_metrics, width, label='Optimized Parameters')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.title('Comparison of Metrics for Subject 10')
plt.tight_layout()
plt.savefig('results/comparison/metrics_comparison.png')
plt.close()

# 2. Side-by-side parameter comparison
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(param_cols))
width = 0.35

# Normalize parameters for better visualization
worst_norm = [(val - min(worst_values)) / (max(worst_values) - min(worst_values)) 
              if max(worst_values) > min(worst_values) else 0.5 for val in worst_values]
best_norm = [(val - min(best_values)) / (max(best_values) - min(best_values)) 
             if max(best_values) > min(best_values) else 0.5 for val in best_values]

ax.bar(x - width/2, worst_norm, width, label='Original Parameters')
ax.bar(x + width/2, best_norm, width, label='Optimized Parameters')
ax.set_xticks(x)
ax.set_xticklabels(param_cols, rotation=45, ha='right')
ax.legend()
plt.title('Normalized Parameter Comparison for Subject 10')
plt.tight_layout()
plt.savefig('results/comparison/parameter_comparison_normalized.png')
plt.close()

# 3. Create a table comparing actual parameter values
comparison_data = {
    'Parameter': param_cols,
    'Original Value': worst_values,
    'Optimized Value': best_values,
    'Difference': best_values - worst_values,
    'Percent Change': ((best_values - worst_values) / worst_values) * 100
}
comparison_df = pd.DataFrame(comparison_data)

# Format the dataframe for better readability
pd.set_option('display.float_format', '{:.2f}'.format)
print("Parameter Comparison for Subject 10:")
print(comparison_df)

# Save to CSV
comparison_df.to_csv('results/comparison/parameter_comparison.csv', index=False)

# 4. Radar chart to compare both parameter sets
# Normalize between 0 and 1 for radar chart
# Use the same normalization for both to enable direct comparison
min_vals = np.min([worst_values, best_values], axis=0)
max_vals = np.max([worst_values, best_values], axis=0)
range_vals = max_vals - min_vals

# Avoid division by zero
range_vals[range_vals == 0] = 1.0

worst_norm = (worst_values - min_vals) / range_vals
best_norm = (best_values - min_vals) / range_vals

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(param_cols), endpoint=False).tolist()
# Close the plot
worst_norm = np.append(worst_norm, worst_norm[0])
best_norm = np.append(best_norm, best_norm[0])
angles.append(angles[0])
param_cols_plot = param_cols + [param_cols[0]]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.fill(angles, worst_norm, color='red', alpha=0.25, label='Original Parameters')
ax.plot(angles, worst_norm, color='red', linewidth=2)
ax.fill(angles, best_norm, color='green', alpha=0.25, label='Optimized Parameters')
ax.plot(angles, best_norm, color='green', linewidth=2)
ax.set_thetagrids(np.degrees(angles[:-1]), param_cols)
ax.set_ylim(0, 1)
plt.title('Parameter Comparison: Original vs Optimized (Subject 10)')
plt.legend(loc='upper right')
plt.savefig('results/comparison/parameter_radar_comparison.png')
plt.close()

# 5. Create a bar chart showing metabolic cost for all parameter configurations
plt.figure(figsize=(10, 6))
configs = np.arange(len(subject_10_df))
metabolic_costs = subject_10_df['metabolic_cost'].values
colors = ['green' if cost == min(metabolic_costs) else 'red' if cost == max(metabolic_costs) else 'gray' 
          for cost in metabolic_costs]

plt.bar(configs, metabolic_costs, color=colors)
plt.axhline(y=metabolic_costs.mean(), color='black', linestyle='--', label='Average')
plt.xlabel('Configuration Number')
plt.ylabel('Metabolic Cost')
plt.title('Metabolic Cost for Different Parameter Configurations (Subject 10)')
plt.legend()
plt.tight_layout()
plt.savefig('results/comparison/metabolic_cost_configurations.png')
plt.close()

# 6. Create a joint load comparison chart
if 'joint_loads' in worst_params and 'joint_loads' in best_params:
    # If we have joint load data in the dataframe
    try:
        worst_loads = eval(worst_params['joint_loads'])
        best_loads = eval(best_params['joint_loads'])
        
        joints = list(worst_loads.keys())
        worst_load_vals = [worst_loads[j] for j in joints]
        best_load_vals = [best_loads[j] for j in joints]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(joints))
        width = 0.35
        
        ax.bar(x - width/2, worst_load_vals, width, label='Original Parameters')
        ax.bar(x + width/2, best_load_vals, width, label='Optimized Parameters')
        ax.set_xticks(x)
        ax.set_xticklabels(joints)
        ax.legend()
        plt.title('Joint Load Comparison for Subject 10')
        plt.ylabel('Joint Load')
        plt.tight_layout()
        plt.savefig('results/comparison/joint_load_comparison.png')
        plt.close()
    except:
        pass  # Skip if joint_loads is not in expected format

print("Parameter comparison analysis complete. Results saved to results/comparison directory.") 