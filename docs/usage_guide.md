# Usage Guide: Personalized Exoskeleton Adaptation using ML and OpenSim

This guide explains how to use the main components of the project for personalized exoskeleton parameter tuning.

## Table of Contents
1. [Installation](#installation)
2. [Data Processing](#data-processing)
3. [OpenSim Integration](#opensim-integration)
4. [Machine Learning Models](#machine-learning-models)
5. [Visualization](#visualization)
6. [End-to-End Example](#end-to-end-example)

## Installation

Follow these steps to set up your environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/personalized-exo-adaptation.git
cd personalized-exo-adaptation

# Create and activate a conda environment
conda env create -f environment.yml
conda activate exo-ml

# Install additional Python packages
pip install -r requirements.txt
```

## Data Processing

The `BiomechanicalDataProcessor` class in `src/data_processing/biomech_data_processor.py` handles loading, preprocessing, and feature extraction from biomechanical data.

```python
from src.data_processing.biomech_data_processor import BiomechanicalDataProcessor

# Initialize processor with data directory
processor = BiomechanicalDataProcessor(data_dir='data/raw')

# Load data for a subject
data = processor.load_motion_data(subject_id=1)

# Preprocess data
processed_data = processor.preprocess_data(data)

# Segment into gait cycles
gait_cycles = processor.segment_gait_cycles(processed_data)

# Extract features
basic_features = processor.extract_features(feature_set='basic')
biomech_features = processor.extract_features(feature_set='biomechanical')
```

## OpenSim Integration

The OpenSim integration modules in `src/opensim_integration/` provide interfaces to run simulations with exoskeleton parameters.

### Exoskeleton Parameters

```python
from src.opensim_integration.exoskeleton_model import ExoskeletonParameters, ExoskeletonController

# Create exoskeleton parameters
exo_params = ExoskeletonParameters()

# Set specific parameters
exo_params.set_parameter('peak_torque', 50.0)
exo_params.set_parameter('onset_percent', 15.0)
exo_params.set_parameter('knee_assist_weight', 0.8)

# Generate random parameters
random_params = exo_params.get_random_parameters()
exo_params.set_parameters_from_dict(random_params)

# Create a controller using the parameters
controller = ExoskeletonController(parameters=exo_params)

# Generate torque profile for a specific point in gait cycle
torques = controller.generate_torque_profile(gait_percent=25.0)

# Generate control trajectory for a full simulation
controls = controller.generate_control_trajectory(duration=1.0)
```

### Running OpenSim Simulations

```python
from src.opensim_integration.opensim_interface import OpenSimInterface

# Initialize OpenSim interface with a model
osim = OpenSimInterface(model_path='data/opensim_models/model.osim')

# Add exoskeleton to the model
osim_params = exo_params.get_opensim_parameters()
osim.add_exoskeleton(osim_params['knee'])

# Define initial state for simulation
motion_data = {
    'initial_positions': {'knee_angle_r': 0.0, 'hip_flexion_r': 0.0},
    'initial_velocities': {'knee_angle_r': 0.0, 'hip_flexion_r': 0.0}
}

# Run simulation
result = osim.run_simulation(motion_data, duration=1.0, output_dir='results')

# Analyze results
metrics = osim.analyze_results(result['result_file'], exo_params=osim_params)
```

## Machine Learning Models

The `ExoParameterPredictor` class in `src/ml_models/exo_parameter_predictor.py` trains and uses machine learning models to predict optimal exoskeleton parameters based on user features.

```python
from src.ml_models.exo_parameter_predictor import ExoParameterPredictor

# Initialize predictor
predictor = ExoParameterPredictor(model_type='random_forest')

# Train model
results = predictor.train(feature_data, target_data)

# Get feature importance
importance = predictor.get_feature_importance()

# Make predictions for new users
predictions = predictor.predict(new_user_features)

# Save and load model
predictor.save_model(model_dir='models')
predictor.load_model(model_dir='models')
```

### Transfer Learning

The `TransferLearningPredictor` class adapts pre-trained models to new users with limited data.

```python
from src.ml_models.exo_parameter_predictor import TransferLearningPredictor

# Initialize with base model
transfer_model = TransferLearningPredictor(base_model_path='models')

# Adapt to a specific user
adaptation_results = transfer_model.adapt_to_user(
    user_features, user_targets, adaptation_strategy='fine_tune'
)

# Make predictions
predictions = transfer_model.predict(user_features)
```

## Visualization

The `ResultVisualizer` class in `src/visualization/result_visualizer.py` creates visualizations of biomechanical data, exoskeleton parameters, and simulation results.

```python
from src.visualization.result_visualizer import ResultVisualizer

# Initialize visualizer
visualizer = ResultVisualizer(output_dir='results/figures')

# Plot joint kinematics
fig = visualizer.plot_gait_kinematics(gait_data, joint_cols=['knee_angle_r', 'hip_angle_r'])
visualizer.save_figure(fig, "joint_kinematics")

# Plot exoskeleton parameters
fig = visualizer.plot_exo_parameters(exo_params.params)
visualizer.save_figure(fig, "exo_parameters")

# Plot torque profiles
controller = ExoskeletonController(parameters=exo_params)
fig = visualizer.plot_torque_profiles(controller)
visualizer.save_figure(fig, "torque_profiles")

# Plot predicted vs actual parameters
fig = visualizer.plot_predicted_vs_actual(actual_params, predicted_params)
visualizer.save_figure(fig, "prediction_accuracy")

# Plot feature importance
fig = visualizer.plot_feature_importance(importance)
visualizer.save_figure(fig, "feature_importance")

# Plot metabolic cost reduction
fig = visualizer.plot_metabolic_cost_reduction(simulation_results)
visualizer.save_figure(fig, "metabolic_cost_reduction")

# Create interactive 3D surface plot
fig = visualizer.plot_interactive_3d_surface(
    simulation_results, 
    x_col='peak_torque', 
    y_col='knee_assist_weight', 
    z_col='metabolic_cost'
)
visualizer.save_figure(fig, "parameter_optimization_surface")
```

## End-to-End Example

The `src/main.py` script demonstrates an end-to-end workflow:

1. Generate sample biomechanical data
2. Process data and extract features
3. Simulate exoskeleton assistance with different parameters
4. Train a machine learning model to predict optimal parameters
5. Personalize parameters for new users
6. Visualize results

To run the example:

```bash
# Make sure you're in the root directory of the project
python src/main.py
```

This will create sample data, run simulations, train models, and generate visualizations in the respective directories.

## Custom Workflows

You can create custom workflows by combining the modules according to your research needs. For example:

1. Use your own biomechanical data by placing it in the `data/raw` directory
2. Create custom OpenSim models and place them in `data/opensim_models`
3. Implement additional feature extraction methods in the data processor
4. Experiment with different machine learning models and hyperparameters
5. Create new visualization functions for specific research questions

See the individual module documentation for more details on available functions and parameters. 