#!/usr/bin/env python3
"""
Custom visualization script for the AI+Biomech exoskeleton parameters data.
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
os.makedirs('results/custom_figures', exist_ok=True)

# Load the simulation results
results_df = pd.read_csv('results/simulation_results.csv')

# 1. Plot the distribution of metabolic cost by subject
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
sns.boxplot(x='subject_id', y='metabolic_cost', data=results_df)
plt.title('Metabolic Cost Distribution by Subject')
plt.xlabel('Subject ID')
plt.ylabel('Metabolic Cost')

plt.subplot(1, 2, 2)
sns.violinplot(x='subject_id', y='metabolic_cost', data=results_df)
plt.title('Metabolic Cost Distribution by Subject (Violin Plot)')
plt.xlabel('Subject ID')
plt.ylabel('Metabolic Cost')
plt.tight_layout()
plt.savefig('results/custom_figures/metabolic_cost_distribution.png')
plt.close()

# 2. Plot assistance efficiency vs metabolic cost
plt.figure(figsize=(10, 6))
sns.scatterplot(x='assistance_efficiency', y='metabolic_cost', hue='subject_id', 
                data=results_df, s=100, alpha=0.7)
plt.title('Assistance Efficiency vs Metabolic Cost')
plt.xlabel('Assistance Efficiency')
plt.ylabel('Metabolic Cost')
plt.grid(True)
plt.savefig('results/custom_figures/efficiency_vs_cost.png')
plt.close()

# 3. Plot heatmap of correlations between parameters
param_cols = ['onset_percent', 'offset_percent', 'peak_torque', 'rise_time', 
              'fall_time', 'stiffness', 'damping', 'knee_assist_weight', 
              'hip_assist_weight', 'ankle_assist_weight']
corr_matrix = results_df[param_cols + ['metabolic_cost', 'assistance_efficiency']].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            square=True, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Parameters')
plt.tight_layout()
plt.savefig('results/custom_figures/parameter_correlations.png')
plt.close()

# 4. Plot parameter distributions
fig, axs = plt.subplots(5, 2, figsize=(14, 18))
axs = axs.flatten()

for i, param in enumerate(param_cols):
    sns.histplot(results_df[param], kde=True, ax=axs[i])
    axs[i].set_title(f'Distribution of {param}')
    axs[i].set_xlabel(param)
    axs[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/custom_figures/parameter_distributions.png')
plt.close()

# 5. Plot pairwise relationships between selected parameters
key_params = ['peak_torque', 'stiffness', 'damping', 'metabolic_cost', 'assistance_efficiency']
sns.pairplot(results_df[key_params], diag_kind='kde', height=2.5)
plt.suptitle('Pairwise Relationships Between Key Parameters', y=1.02)
plt.savefig('results/custom_figures/parameter_pairplots.png')
plt.close()

# 6. Create parallel coordinates plot for the best parameters per subject
# First, find the minimum metabolic cost configuration for each subject
best_params = results_df.loc[results_df.groupby('subject_id')['metabolic_cost'].idxmin()]

# Normalize data for parallel plot
normalized_best = best_params.copy()
for col in param_cols:
    normalized_best[col] = (normalized_best[col] - normalized_best[col].min()) / (normalized_best[col].max() - normalized_best[col].min())

plt.figure(figsize=(14, 8))
pd.plotting.parallel_coordinates(normalized_best, 'subject_id', 
                                 colormap=plt.cm.tab10, 
                                 cols=param_cols)
plt.title('Optimal Exoskeleton Parameters by Subject (Normalized)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('results/custom_figures/optimal_parameters_parallel.png')
plt.close()

# 7. Plot subject demographics vs average metabolic cost
subject_avg = results_df.groupby('subject_id')[['height', 'weight', 'age', 'metabolic_cost']].mean().reset_index()

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
sns.scatterplot(x='height', y='metabolic_cost', data=subject_avg, s=100, ax=axs[0])
axs[0].set_title('Height vs Average Metabolic Cost')

sns.scatterplot(x='weight', y='metabolic_cost', data=subject_avg, s=100, ax=axs[1])
axs[1].set_title('Weight vs Average Metabolic Cost')

sns.scatterplot(x='age', y='metabolic_cost', data=subject_avg, s=100, ax=axs[2])
axs[2].set_title('Age vs Average Metabolic Cost')

plt.tight_layout()
plt.savefig('results/custom_figures/demographics_vs_cost.png')
plt.close()

# 8. Create a radar chart for the best parameter configuration for subject 10
# Prepare data for radar chart
subject_10_best = best_params[best_params['subject_id'] == 10]
if not subject_10_best.empty:
    # Normalize between 0 and 1 for radar chart
    radar_data = subject_10_best[param_cols].iloc[0].tolist()
    # Calculate min and max for each parameter
    min_vals = results_df[param_cols].min().values
    max_vals = results_df[param_cols].max().values
    # Normalize
    radar_data_norm = [(val - min_val) / (max_val - min_val) if max_val > min_val else 0.5 
                      for val, min_val, max_val in zip(radar_data, min_vals, max_vals)]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(param_cols), endpoint=False).tolist()
    # Close the plot
    radar_data_norm.append(radar_data_norm[0])
    angles.append(angles[0])
    param_cols_plot = param_cols + [param_cols[0]]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.fill(angles, radar_data_norm, color='skyblue', alpha=0.5)
    ax.plot(angles, radar_data_norm, color='blue', linewidth=2)
    ax.set_thetagrids(np.degrees(angles[:-1]), param_cols)
    ax.set_ylim(0, 1)
    plt.title('Optimal Parameters for Subject 10')
    plt.savefig('results/custom_figures/subject_10_radar.png')
    plt.close()

print("Custom visualizations generated and saved to results/custom_figures directory.") 