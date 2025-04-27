"""
Visualization module for biomechanical data and exoskeleton parameters.
This module provides functions to visualize data, model predictions, and simulation results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px


class ResultVisualizer:
    """
    Class for visualizing exoskeleton adaptation results.
    """
    
    def __init__(self, output_dir="results/figures"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save output figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_gait_kinematics(self, data, joint_cols=None, title="Joint Kinematics During Gait Cycle"):
        """
        Plot joint angles during the gait cycle.
        
        Args:
            data (DataFrame): Motion data with percent_gait column
            joint_cols (list, optional): Joint angle columns to plot
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        if 'percent_gait' not in data.columns:
            raise ValueError("Data must include percent_gait column")
        
        if joint_cols is None:
            # Find joint angle columns
            joint_cols = [col for col in data.columns if '_angle_' in col]
            if not joint_cols:
                raise ValueError("No joint angle columns found in data")
        
        # Create figure with subplots for each joint
        fig, axes = plt.subplots(len(joint_cols), 1, figsize=(10, 2+2*len(joint_cols)), sharex=True)
        if len(joint_cols) == 1:
            axes = [axes]
        
        # Plot each joint
        for i, col in enumerate(joint_cols):
            ax = axes[i]
            ax.plot(data['percent_gait'], data[col])
            ax.set_ylabel(col.replace('_', ' ').capitalize())
            ax.grid(True)
        
        axes[-1].set_xlabel('Gait Cycle (%)')
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_exo_parameters(self, parameters, title="Exoskeleton Parameters"):
        """
        Visualize exoskeleton parameters.
        
        Args:
            parameters (dict): Dictionary of parameter values
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(parameters, dict):
            df = pd.DataFrame([parameters])
        else:
            df = parameters
        
        # Sort parameters by value
        param_names = list(df.columns)
        param_values = df.iloc[0].values
        
        # Sort by value
        sorted_indices = np.argsort(param_values)
        sorted_names = [param_names[i] for i in sorted_indices]
        sorted_values = [param_values[i] for i in sorted_indices]
        
        # Create bar chart
        bars = ax.barh(sorted_names, sorted_values)
        
        # Add values to the end of bars
        for i, v in enumerate(sorted_values):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center')
        
        ax.set_xlabel('Parameter Value')
        ax.set_title(title)
        ax.grid(True, axis='x')
        
        return fig
    
    def plot_torque_profiles(self, controller, title="Exoskeleton Torque Profiles"):
        """
        Plot torque profiles for different joints over the gait cycle.
        
        Args:
            controller (ExoskeletonController): Controller with parameters
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        # Generate data for the full gait cycle
        gait_percents = np.linspace(0, 100, 101)
        joint_torques = {
            'knee': [],
            'hip': [],
            'ankle': []
        }
        
        # Get torque values at each point in gait cycle
        for percent in gait_percents:
            torques = controller.generate_torque_profile(percent)
            for joint in joint_torques:
                joint_torques[joint].append(torques[joint])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot torque profiles
        ax.plot(gait_percents, joint_torques['knee'], label='Knee')
        ax.plot(gait_percents, joint_torques['hip'], label='Hip')
        ax.plot(gait_percents, joint_torques['ankle'], label='Ankle')
        
        ax.set_xlabel('Gait Cycle (%)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_predicted_vs_actual(self, actual, predicted, param_names=None, title="Predicted vs Actual Parameters"):
        """
        Plot predicted vs actual exoskeleton parameters.
        
        Args:
            actual (DataFrame): Actual parameters
            predicted (DataFrame): Predicted parameters
            param_names (list, optional): Parameter names to include
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        if param_names is None:
            param_names = actual.columns
        
        # Number of parameters to plot
        n_params = len(param_names)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
        if n_params == 1:
            axes = [axes]
        
        # Plot each parameter
        for i, param in enumerate(param_names):
            ax = axes[i]
            
            # Get actual and predicted values
            actual_values = actual[param].values
            predicted_values = predicted[param].values
            
            # Calculate correlation coefficient
            corr = np.corrcoef(actual_values, predicted_values)[0, 1]
            
            # Scatter plot
            ax.scatter(actual_values, predicted_values, alpha=0.7)
            
            # Add diagonal line (perfect prediction)
            min_val = min(min(actual_values), min(predicted_values))
            max_val = max(max(actual_values), max(predicted_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--')
            
            # Add correlation coefficient
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(param.replace('_', ' ').capitalize())
            ax.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, feature_importance, title="Feature Importance for Parameter Prediction"):
        """
        Plot feature importance from ML model.
        
        Args:
            feature_importance (DataFrame): Feature importance data
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        # Sort by importance
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars
        bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'])
        
        # Add values
        for i, v in enumerate(feature_importance['Importance']):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, axis='x')
        
        # Limit to top 20 features if there are many
        if len(feature_importance) > 20:
            plt.ylim(-0.5, 19.5)
            ax.set_yticks(range(20))
            ax.set_yticklabels(feature_importance['Feature'].values[:20])
        
        return fig
    
    def plot_metabolic_cost_reduction(self, sim_results, title="Metabolic Cost Reduction with Exoskeleton"):
        """
        Plot metabolic cost reduction with different exoskeleton parameters.
        
        Args:
            sim_results (DataFrame): Simulation results containing metabolic cost
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot metabolic cost against key parameter(s)
        if 'peak_torque' in sim_results.columns:
            sns.scatterplot(
                x='peak_torque', 
                y='metabolic_cost', 
                hue='subject_id' if 'subject_id' in sim_results.columns else None,
                size='assistance_efficiency' if 'assistance_efficiency' in sim_results.columns else None,
                data=sim_results,
                ax=ax
            )
            ax.set_xlabel('Peak Torque (Nm)')
            
        elif 'knee_assist_weight' in sim_results.columns and 'hip_assist_weight' in sim_results.columns:
            # For multiple parameters, create a heatmap or 3D plot
            pivot = sim_results.pivot_table(
                values='metabolic_cost',
                index='knee_assist_weight',
                columns='hip_assist_weight'
            )
            
            sns.heatmap(pivot, annot=True, cmap='viridis_r', ax=ax)
            ax.set_xlabel('Hip Assistance Weight')
            ax.set_ylabel('Knee Assistance Weight')
        
        ax.set_ylabel('Metabolic Cost (W/kg)')
        ax.set_title(title)
        
        return fig
    
    def plot_subject_adaptation(self, subject_results, title="Subject-Specific Adaptation"):
        """
        Plot adaptation results for different subjects.
        
        Args:
            subject_results (DataFrame): Results for different subjects
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        if 'subject_id' not in subject_results.columns:
            raise ValueError("subject_results must contain 'subject_id' column")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot 1: Metabolic cost by subject
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(x='subject_id', y='metabolic_cost', data=subject_results, ax=ax1)
        ax1.set_xlabel('Subject ID')
        ax1.set_ylabel('Metabolic Cost (W/kg)')
        ax1.set_title('Metabolic Cost by Subject')
        ax1.grid(True, axis='y')
        
        # Plot 2: Optimal parameters by subject
        param_cols = [col for col in subject_results.columns 
                    if any(s in col for s in ['torque', 'onset', 'offset', 'weight'])]
        
        if param_cols:
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Reshape data for plotting
            melted = pd.melt(subject_results, id_vars=['subject_id'], value_vars=param_cols,
                            var_name='Parameter', value_name='Value')
            
            sns.boxplot(x='Parameter', y='Value', data=melted, ax=ax2)
            ax2.set_xlabel('Parameter')
            ax2.set_ylabel('Value')
            ax2.set_title('Parameter Variation Across Subjects')
            ax2.grid(True, axis='y')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 3: Subject characteristics vs optimal parameters
        if 'height' in subject_results.columns and 'weight' in subject_results.columns:
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Create a derived measure (e.g., BMI)
            subject_results['bmi'] = subject_results['weight'] / (subject_results['height']**2)
            
            # Choose a key parameter to plot against BMI
            key_param = param_cols[0] if param_cols else 'metabolic_cost'
            
            sns.scatterplot(
                x='bmi', 
                y=key_param, 
                hue='subject_id', 
                size='metabolic_cost' if key_param != 'metabolic_cost' else None,
                data=subject_results,
                ax=ax3
            )
            ax3.set_xlabel('BMI (kg/m²)')
            ax3.set_ylabel(key_param.replace('_', ' ').capitalize())
            ax3.set_title('Subject Characteristics vs Parameters')
            ax3.grid(True)
        
        # Plot 4: Assistance efficiency
        if 'assistance_efficiency' in subject_results.columns:
            ax4 = fig.add_subplot(gs[1, 1])
            
            if 'peak_torque' in subject_results.columns:
                sns.scatterplot(
                    x='peak_torque', 
                    y='assistance_efficiency', 
                    hue='subject_id',
                    data=subject_results,
                    ax=ax4
                )
                ax4.set_xlabel('Peak Torque (Nm)')
            else:
                sns.barplot(x='subject_id', y='assistance_efficiency', data=subject_results, ax=ax4)
                ax4.set_xlabel('Subject ID')
            
            ax4.set_ylabel('Assistance Efficiency')
            ax4.set_title('Exoskeleton Assistance Efficiency')
            ax4.grid(True)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_interactive_3d_surface(self, sim_results, x_col, y_col, z_col, title="Parameter Optimization Surface"):
        """
        Create an interactive 3D surface plot of simulation results.
        
        Args:
            sim_results (DataFrame): Simulation results
            x_col (str): Column for x-axis
            y_col (str): Column for y-axis
            z_col (str): Column for z-axis (usually a metric)
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive 3D figure
        """
        # Create pivot table for surface data
        surface_data = sim_results.pivot_table(
            values=z_col,
            index=x_col,
            columns=y_col
        )
        
        # Get axis values
        x_values = surface_data.index.values
        y_values = surface_data.columns.values
        z_values = surface_data.values
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(z=z_values, x=x_values, y=y_values)])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col.replace('_', ' ').capitalize(),
                yaxis_title=y_col.replace('_', ' ').capitalize(),
                zaxis_title=z_col.replace('_', ' ').capitalize()
            ),
            width=800,
            height=700
        )
        
        return fig
    
    def save_figure(self, fig, filename, formats=None):
        """
        Save a figure to file(s).
        
        Args:
            fig: Figure object (matplotlib or plotly)
            filename (str): Base filename without extension
            formats (list, optional): List of formats to save
            
        Returns:
            list: Paths to saved files
        """
        if formats is None:
            formats = ['png', 'pdf']
        
        saved_paths = []
        
        for fmt in formats:
            output_path = os.path.join(self.output_dir, f"{filename}.{fmt}")
            
            if hasattr(fig, 'write_image'):  # Plotly figure
                fig.write_image(output_path)
            else:  # Matplotlib figure
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
            
            saved_paths.append(output_path)
        
        return saved_paths 