"""
Main script to demonstrate the integration of OpenSim and machine learning
for personalized exoskeleton parameter tuning.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.opensim_integration.opensim_interface import OpenSimInterface
from src.opensim_integration.exoskeleton_model import ExoskeletonParameters, ExoskeletonController
from src.data_processing.biomech_data_processor import BiomechanicalDataProcessor
from src.ml_models.exo_parameter_predictor import ExoParameterPredictor, TransferLearningPredictor
from src.visualization.result_visualizer import ResultVisualizer


def generate_sample_data(n_subjects=10, output_dir='data/raw'):
    """
    Generate sample biomechanical data for demonstration.
    
    Args:
        n_subjects (int): Number of subjects to generate
        output_dir (str): Directory to save data
        
    Returns:
        list: Paths to generated data files
    """
    print(f"Generating sample data for {n_subjects} subjects...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters for data generation
    sampling_rate = 100  # Hz
    gait_cycles = 3
    duration = gait_cycles * 1.0  # seconds
    time_points = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Subject characteristics (height, weight, etc.)
    subject_chars = {
        'height': np.random.normal(1.75, 0.1, n_subjects),  # meters
        'weight': np.random.normal(70, 10, n_subjects),     # kg
        'age': np.random.normal(30, 5, n_subjects),         # years
        'gender': np.random.choice(['M', 'F'], n_subjects)  # M/F
    }
    
    # Generate data for each subject
    file_paths = []
    
    for subject_id in range(1, n_subjects + 1):
        # Create dataframe for this subject
        data = pd.DataFrame()
        data['time'] = time_points
        
        # Add subject metadata
        data['subject_id'] = subject_id
        
        # Generate joint angles with individual variations
        # Hip
        base_freq = 1.0  # 1 Hz gait cycle
        phase_shift = np.random.uniform(0, 0.1)
        amplitude = np.random.uniform(0.9, 1.1)
        
        # Hip flexion/extension
        data['hip_angle_r'] = amplitude * 30 * np.sin(2 * np.pi * base_freq * time_points + phase_shift)
        data['hip_angle_l'] = amplitude * 30 * np.sin(2 * np.pi * base_freq * time_points + phase_shift + np.pi)
        
        # Knee flexion/extension
        k_amp = np.random.uniform(0.9, 1.1) * 60
        data['knee_angle_r'] = k_amp * np.abs(np.sin(2 * np.pi * base_freq * time_points + phase_shift))
        data['knee_angle_l'] = k_amp * np.abs(np.sin(2 * np.pi * base_freq * time_points + phase_shift + np.pi))
        
        # Ankle dorsi/plantarflexion
        a_amp = np.random.uniform(0.9, 1.1) * 20
        a_phase = np.random.uniform(-0.1, 0.1)
        data['ankle_angle_r'] = a_amp * np.sin(2 * np.pi * base_freq * time_points + phase_shift + a_phase)
        data['ankle_angle_l'] = a_amp * np.sin(2 * np.pi * base_freq * time_points + phase_shift + np.pi + a_phase)
        
        # Add marker positions
        data['heel_marker_r_z'] = 0.05 + 0.05 * np.abs(np.sin(2 * np.pi * base_freq * time_points + phase_shift))
        data['heel_marker_l_z'] = 0.05 + 0.05 * np.abs(np.sin(2 * np.pi * base_freq * time_points + phase_shift + np.pi))
        
        # Save to CSV
        file_path = os.path.join(output_dir, f"subject_{subject_id}.csv")
        data.to_csv(file_path, index=False)
        file_paths.append(file_path)
        
        print(f"Generated data for subject {subject_id}: {file_path}")
    
    # Save subject characteristics
    subject_df = pd.DataFrame({
        'subject_id': range(1, n_subjects + 1),
        'height': subject_chars['height'],
        'weight': subject_chars['weight'],
        'age': subject_chars['age'],
        'gender': subject_chars['gender']
    })
    
    subject_file = os.path.join(output_dir, "subject_characteristics.csv")
    subject_df.to_csv(subject_file, index=False)
    
    return file_paths


def process_subject_data(subject_id, data_dir='data/raw', output_dir='data/processed'):
    """
    Process biomechanical data for a single subject.
    
    Args:
        subject_id (int): Subject ID
        data_dir (str): Input data directory
        output_dir (str): Output directory
        
    Returns:
        dict: Processed data and features
    """
    print(f"Processing data for subject {subject_id}...")
    
    # Initialize data processor
    processor = BiomechanicalDataProcessor(data_dir=data_dir)
    
    # Load and process data
    try:
        # Load raw data
        raw_data = processor.load_motion_data(subject_id=subject_id)
        
        # Preprocess data
        processed_data = processor.preprocess_data(raw_data)
        
        # Segment into gait cycles
        gait_cycles = processor.segment_gait_cycles(processed_data)
        
        # Extract features
        basic_features = processor.extract_features(feature_set='basic')
        biomech_features = processor.extract_features(feature_set='biomechanical')
        temporal_features = processor.extract_features(feature_set='temporal')
        
        # Save processed data
        output_file = processor.save_processed_data(
            output_dir=output_dir,
            subject_id=subject_id
        )
        
        # Create a visualization
        visualizer = ResultVisualizer(output_dir=os.path.join(output_dir, 'figures'))
        if len(gait_cycles) > 0:
            joint_cols = [col for col in gait_cycles[0].columns if '_angle_' in col]
            if joint_cols:
                fig = visualizer.plot_gait_kinematics(
                    gait_cycles[0],
                    joint_cols=joint_cols,
                    title=f"Subject {subject_id} - Joint Kinematics"
                )
                visualizer.save_figure(fig, f"subject_{subject_id}_kinematics")
        
        return {
            'raw_data': raw_data,
            'processed_data': processed_data,
            'gait_cycles': gait_cycles,
            'basic_features': basic_features,
            'biomech_features': biomech_features,
            'temporal_features': temporal_features,
            'output_file': output_file
        }
    
    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
        return None


def run_opensim_simulation(subject_id, exo_params, model_path=None, output_dir='results'):
    """
    Run an OpenSim simulation for a subject with specified exoskeleton parameters.
    
    Args:
        subject_id (int): Subject ID
        exo_params (ExoskeletonParameters): Exoskeleton parameters
        model_path (str, optional): Path to OpenSim model
        output_dir (str): Output directory
        
    Returns:
        dict: Simulation results
    """
    print(f"Running OpenSim simulation for subject {subject_id}...")
    
    # Initialize OpenSim interface
    osim = OpenSimInterface(model_path=model_path)
    
    # If no model specified, create a placeholder
    if model_path is None or not osim.model:
        # In a real scenario, we would load a real model
        # Here we just create a placeholder for demonstration
        print("No valid OpenSim model provided, creating placeholder results")
        
        # Get OpenSim parameters
        opensim_params = exo_params.get_opensim_parameters()
        
        # Create controller with the parameters
        controller = ExoskeletonController(parameters=exo_params)
        
        # Generate a control trajectory
        controls = controller.generate_control_trajectory(duration=1.0)
        
        # Simulate metabolic cost reduction based on parameters
        # In a real scenario, this would come from actual simulation
        knee_factor = opensim_params['knee']['max_torque'] / 100.0
        hip_factor = opensim_params['hip']['max_torque'] / 100.0
        ankle_factor = opensim_params['ankle']['max_torque'] / 100.0
        
        # Simple model of metabolic cost reduction (placeholders)
        base_metabolic_cost = 300  # W
        reduction_factor = (0.5 * knee_factor + 0.3 * hip_factor + 0.2 * ankle_factor) * \
                           (1.0 - abs(opensim_params['knee']['onset'] - 0.1) / 0.3)
        
        # Clip reduction to reasonable range
        reduction_factor = min(max(reduction_factor, 0), 0.3)
        metabolic_cost = base_metabolic_cost * (1.0 - reduction_factor)
        
        # Joint loads (placeholders)
        joint_loads = {
            'knee': 50.0 * (1.0 - 0.6 * knee_factor),
            'hip': 70.0 * (1.0 - 0.5 * hip_factor),
            'ankle': 40.0 * (1.0 - 0.4 * ankle_factor)
        }
        
        # Assistance efficiency (placeholder)
        assistance_efficiency = 0.8 - 0.3 * (knee_factor + hip_factor + ankle_factor) / 3.0
        
        # Return simulated results
        return {
            'subject_id': subject_id,
            'metabolic_cost': metabolic_cost,
            'joint_loads': joint_loads,
            'assistance_efficiency': assistance_efficiency,
            'controls': controls,
            'parameters': exo_params.params
        }
    
    else:
        # This would be the actual OpenSim simulation path
        # Here we just return None to indicate it's not implemented
        # In a real scenario, we would:
        # 1. Add exoskeleton to the model
        # 2. Configure simulation with the parameters
        # 3. Run the simulation
        # 4. Process and return results
        print("Full OpenSim simulation not implemented in this demo")
        return None


def train_ml_model(feature_data, target_data, model_type='random_forest', output_dir='models'):
    """
    Train a machine learning model to predict optimal exoskeleton parameters.
    
    Args:
        feature_data (DataFrame): Subject features
        target_data (DataFrame): Optimal exoskeleton parameters
        model_type (str): Type of ML model
        output_dir (str): Output directory
        
    Returns:
        ExoParameterPredictor: Trained model
    """
    print(f"Training {model_type} model for parameter prediction...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    predictor = ExoParameterPredictor(model_type=model_type)
    
    # Train model
    results = predictor.train(feature_data, target_data)
    
    # Print results
    print(f"Training complete. Overall MSE: {results['overall_mse']:.4f}, R²: {results['overall_r2']:.4f}")
    for param, metrics in results['parameter_metrics'].items():
        print(f"  {param}: MSE = {metrics['mse']:.4f}, R² = {metrics['r2']:.4f}")
    
    # Save model
    predictor.save_model(model_dir=output_dir)
    
    # Get feature importance
    if model_type == 'random_forest':
        importance = predictor.get_feature_importance()
        
        # Visualize feature importance
        visualizer = ResultVisualizer(output_dir=os.path.join(output_dir, 'figures'))
        fig = visualizer.plot_feature_importance(importance)
        visualizer.save_figure(fig, "feature_importance")
    
    return predictor


def personalize_for_new_subject(subject_id, base_model_path, data_dir='data/raw', output_dir='results'):
    """
    Personalize exoskeleton parameters for a new subject.
    
    Args:
        subject_id (int): Subject ID
        base_model_path (str): Path to base ML model
        data_dir (str): Input data directory
        output_dir (str): Output directory
        
    Returns:
        dict: Personalization results
    """
    print(f"Personalizing parameters for subject {subject_id}...")
    
    # Process subject data
    subject_data = process_subject_data(subject_id, data_dir)
    if subject_data is None:
        print(f"Could not process data for subject {subject_id}")
        return None
    
    # Create a transfer learning model
    transfer_model = TransferLearningPredictor(base_model_path=base_model_path)
    
    # Extract features for the subject
    features = pd.concat([
        subject_data['basic_features'], 
        subject_data['biomech_features']
    ], axis=1)
    
    # Load subject characteristics
    chars_file = os.path.join(data_dir, "subject_characteristics.csv")
    if os.path.exists(chars_file):
        chars = pd.read_csv(chars_file)
        subject_chars = chars[chars['subject_id'] == subject_id]
        if not subject_chars.empty:
            # Add characteristics to features
            for col in ['height', 'weight', 'age']:
                if col in subject_chars.columns:
                    features[col] = subject_chars[col].values[0]
    
    # Predict initial parameters
    # In a real system, we'd have some optimal target parameters for adaptation
    # Here we'll just create some dummy values
    
    # Create exoskeleton parameters
    exo_params = ExoskeletonParameters()
    
    # Set some random parameters for demonstration
    rand_params = exo_params.get_random_parameters()
    exo_params.set_parameters_from_dict(rand_params)
    
    # Create mock optimal parameters (targets for adaptation)
    optimal_params = ExoskeletonParameters()
    # Modify the random parameters slightly
    for param, value in rand_params.items():
        optimal_val = value * np.random.uniform(0.8, 1.2)  # +/- 20%
        min_val, max_val = exo_params.param_ranges[param]
        optimal_val = min(max(optimal_val, min_val), max_val)  # Keep in range
        optimal_params.set_parameter(param, optimal_val)
    
    # Create dataframes for adaptation
    features_df = features.copy()
    targets_df = pd.DataFrame([optimal_params.params])
    
    # Adapt the model to the user
    adaptation_results = transfer_model.adapt_to_user(
        features_df, targets_df, adaptation_strategy='fine_tune'
    )
    
    print(f"Adaptation results: MSE = {adaptation_results['mse']:.4f}, R² = {adaptation_results['r2']:.4f}")
    
    # Predict personalized parameters
    personalized_params = transfer_model.predict(features_df)
    
    # Create ExoskeletonParameters from predictions
    personalized_exo = ExoskeletonParameters()
    personalized_exo.set_parameters_from_dict(personalized_params.iloc[0].to_dict())
    
    # Run simulation with personalized parameters
    sim_results = run_opensim_simulation(subject_id, personalized_exo)
    
    # Visualize results
    visualizer = ResultVisualizer(output_dir=os.path.join(output_dir, 'figures'))
    
    # Visualize predicted vs actual parameters
    fig = visualizer.plot_predicted_vs_actual(
        targets_df, personalized_params,
        title=f"Subject {subject_id} - Predicted vs Optimal Parameters"
    )
    visualizer.save_figure(fig, f"subject_{subject_id}_prediction")
    
    # Visualize torque profiles
    controller = ExoskeletonController(parameters=personalized_exo)
    fig = visualizer.plot_torque_profiles(
        controller,
        title=f"Subject {subject_id} - Personalized Torque Profiles"
    )
    visualizer.save_figure(fig, f"subject_{subject_id}_torque_profiles")
    
    return {
        'subject_id': subject_id,
        'features': features_df,
        'predicted_params': personalized_params,
        'simulation_results': sim_results
    }


def main():
    """Main entry point for the demonstration."""
    print("Starting personalized exoskeleton adaptation demonstration...")
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/opensim_models', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # STEP 1: Generate sample data
    n_subjects = 10
    data_files = generate_sample_data(n_subjects=n_subjects)
    
    # STEP 2: Process data for each subject
    subject_data_dict = {}
    for subject_id in range(1, n_subjects+1):
        subject_data = process_subject_data(subject_id)
        if subject_data:
            subject_data_dict[subject_id] = subject_data
    
    # STEP 3: Generate simulation results with different exoskeleton parameters
    print("\nGenerating simulation results with different parameters...")
    
    # Create a results DataFrame
    simulation_results = []
    feature_data = []
    target_data = []
    
    # Load subject characteristics
    subject_chars = pd.read_csv('data/raw/subject_characteristics.csv')
    
    # For each subject, try different exoskeleton parameters
    for subject_id, subject_data in subject_data_dict.items():
        print(f"\nExploring parameters for subject {subject_id}...")
        
        # Get subject characteristics
        subject_row = subject_chars[subject_chars['subject_id'] == subject_id]
        
        # Combine all features
        if not subject_data['basic_features'].empty and not subject_data['biomech_features'].empty:
            combined_features = pd.concat([
                subject_data['basic_features'], 
                subject_data['biomech_features']
            ], axis=1)
            
            # Add subject characteristics
            for col in ['height', 'weight', 'age']:
                if col in subject_row.columns:
                    combined_features[col] = subject_row[col].values[0]
            
            # Try different parameter sets
            best_metabolic_cost = float('inf')
            best_params = None
            
            for i in range(5):  # Try 5 different parameter sets
                # Create random parameters
                exo_params = ExoskeletonParameters()
                rand_params = exo_params.get_random_parameters()
                exo_params.set_parameters_from_dict(rand_params)
                
                # Run simulation
                sim_result = run_opensim_simulation(subject_id, exo_params)
                
                if sim_result:
                    # Track best parameters
                    if sim_result['metabolic_cost'] < best_metabolic_cost:
                        best_metabolic_cost = sim_result['metabolic_cost']
                        best_params = exo_params.params.copy()
                    
                    # Add to results
                    row = {
                        'subject_id': subject_id,
                        'height': subject_row['height'].values[0] if 'height' in subject_row else 0,
                        'weight': subject_row['weight'].values[0] if 'weight' in subject_row else 0,
                        'age': subject_row['age'].values[0] if 'age' in subject_row else 0,
                        'metabolic_cost': sim_result['metabolic_cost'],
                        'assistance_efficiency': sim_result['assistance_efficiency']
                    }
                    row.update(exo_params.params)
                    simulation_results.append(row)
            
            # For the best parameters, save features and targets for ML
            if best_params:
                feature_data.append(combined_features)
                target_data.append(pd.DataFrame([best_params]))
    
    # Convert to DataFrames
    simulation_df = pd.DataFrame(simulation_results)
    
    # Save simulation results
    simulation_df.to_csv('results/simulation_results.csv', index=False)
    
    # STEP 4: Train ML model to predict optimal parameters
    if feature_data and target_data:
        features_df = pd.concat(feature_data).reset_index(drop=True)
        targets_df = pd.concat(target_data).reset_index(drop=True)
        
        # Train model
        ml_model = train_ml_model(features_df, targets_df)
        
        # STEP 5: Personalize for a new subject
        # Use the last subject as a "new" subject
        new_subject_id = n_subjects
        personalization = personalize_for_new_subject(
            new_subject_id, 
            base_model_path='models'
        )
        
        if personalization:
            print(f"\nPersonalization complete for subject {new_subject_id}")
            print(f"Predicted parameters:")
            print(personalization['predicted_params'])
            print("\nSimulation results:")
            for key, value in personalization['simulation_results'].items():
                if key != 'controls' and key != 'parameters':
                    print(f"  {key}: {value}")
    
    # STEP 6: Visualize overall results
    print("\nGenerating summary visualizations...")
    visualizer = ResultVisualizer(output_dir='results/figures')
    
    # Plot metabolic cost reduction
    if not simulation_df.empty:
        fig = visualizer.plot_metabolic_cost_reduction(
            simulation_df,
            title="Metabolic Cost vs. Peak Torque"
        )
        visualizer.save_figure(fig, "metabolic_cost_reduction")
        
        # Plot subject adaptation
        fig = visualizer.plot_subject_adaptation(
            simulation_df,
            title="Subject-Specific Exoskeleton Adaptation"
        )
        visualizer.save_figure(fig, "subject_adaptation_summary")
    
    print("\nDemonstration complete. Results saved in results/ directory.")


if __name__ == "__main__":
    main() 