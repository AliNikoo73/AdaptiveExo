"""
Exoskeleton model definitions and parameter management.
This module defines the exoskeleton model and parameter space for personalization.
"""

import numpy as np


class ExoskeletonParameters:
    """Class to manage exoskeleton parameters and their valid ranges."""
    
    # Default parameter ranges
    DEFAULT_RANGES = {
        # Timing parameters
        'onset_percent': (0, 50),   # % of gait cycle
        'offset_percent': (50, 100), # % of gait cycle
        
        # Torque parameters 
        'peak_torque': (0, 100),    # Nm
        'rise_time': (0.05, 0.5),   # seconds
        'fall_time': (0.05, 0.5),   # seconds
        
        # Control parameters
        'stiffness': (0, 100),      # Nm/rad
        'damping': (0, 10),         # Nms/rad
        
        # Joint-specific parameters
        'knee_assist_weight': (0, 1),
        'hip_assist_weight': (0, 1),
        'ankle_assist_weight': (0, 1)
    }
    
    def __init__(self, param_ranges=None):
        """
        Initialize exoskeleton parameters with optional custom ranges.
        
        Args:
            param_ranges (dict, optional): Custom parameter ranges to override defaults
        """
        self.param_ranges = self.DEFAULT_RANGES.copy()
        if param_ranges:
            self.param_ranges.update(param_ranges)
        
        # Initialize parameters to middle of ranges
        self.params = {k: (v[0] + v[1]) / 2.0 for k, v in self.param_ranges.items()}
    
    def set_parameter(self, name, value):
        """
        Set a parameter to a value, ensuring it's within valid range.
        
        Args:
            name (str): Parameter name
            value (float): Parameter value
            
        Returns:
            bool: Success status
        """
        if name not in self.param_ranges:
            print(f"Unknown parameter: {name}")
            return False
        
        min_val, max_val = self.param_ranges[name]
        if value < min_val or value > max_val:
            print(f"Value {value} for {name} out of range [{min_val}, {max_val}]")
            return False
        
        self.params[name] = value
        return True
    
    def get_parameter(self, name):
        """
        Get the current value of a parameter.
        
        Args:
            name (str): Parameter name
            
        Returns:
            float: Parameter value or None if not found
        """
        return self.params.get(name)
    
    def get_opensim_parameters(self):
        """
        Convert parameters to OpenSim-compatible format.
        
        Returns:
            dict: Parameters formatted for OpenSim
        """
        # Map internal parameters to OpenSim model parameters
        opensim_params = {
            'knee': {
                'joint': 'knee_angle_r',
                'max_torque': self.params['peak_torque'] * self.params['knee_assist_weight'],
                'min_control': 0.0,
                'max_control': 1.0,
                'onset': self.params['onset_percent'] / 100.0,
                'offset': self.params['offset_percent'] / 100.0
            },
            'hip': {
                'joint': 'hip_flexion_r',
                'max_torque': self.params['peak_torque'] * self.params['hip_assist_weight'],
                'min_control': 0.0,
                'max_control': 1.0,
                'onset': self.params['onset_percent'] / 100.0,
                'offset': self.params['offset_percent'] / 100.0
            },
            'ankle': {
                'joint': 'ankle_angle_r',
                'max_torque': self.params['peak_torque'] * self.params['ankle_assist_weight'],
                'min_control': 0.0,
                'max_control': 1.0,
                'onset': self.params['onset_percent'] / 100.0,
                'offset': self.params['offset_percent'] / 100.0
            }
        }
        
        return opensim_params
    
    def get_random_parameters(self):
        """
        Generate a random set of parameters within the valid ranges.
        
        Returns:
            dict: Random parameter set
        """
        rand_params = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            rand_params[param] = min_val + np.random.random() * (max_val - min_val)
        
        return rand_params
    
    def set_parameters_from_dict(self, param_dict):
        """
        Set multiple parameters from a dictionary.
        
        Args:
            param_dict (dict): Dictionary of parameter name-value pairs
            
        Returns:
            bool: Success status
        """
        success = True
        for name, value in param_dict.items():
            if not self.set_parameter(name, value):
                success = False
        
        return success


class ExoskeletonController:
    """Class to generate control signals for the exoskeleton."""
    
    def __init__(self, parameters=None):
        """
        Initialize the controller with parameters.
        
        Args:
            parameters (ExoskeletonParameters, optional): Parameters for the controller
        """
        self.parameters = parameters or ExoskeletonParameters()
    
    def generate_torque_profile(self, gait_cycle_percent):
        """
        Generate the torque profile at a given point in the gait cycle.
        
        Args:
            gait_cycle_percent (float): Current percent through gait cycle (0-100)
            
        Returns:
            dict: Torque values for each joint
        """
        # Get parameters
        onset = self.parameters.get_parameter('onset_percent')
        offset = self.parameters.get_parameter('offset_percent')
        peak = self.parameters.get_parameter('peak_torque')
        
        # Initialize torques
        torques = {
            'knee': 0.0,
            'hip': 0.0,
            'ankle': 0.0
        }
        
        # Simple control logic: apply torque between onset and offset
        if onset <= gait_cycle_percent <= offset:
            # Simple trapezoid profile
            if gait_cycle_percent < onset + 5:
                # Ramp up
                pct = (gait_cycle_percent - onset) / 5.0
                scale = pct
            elif gait_cycle_percent > offset - 5:
                # Ramp down
                pct = (offset - gait_cycle_percent) / 5.0
                scale = pct
            else:
                # Full torque
                scale = 1.0
            
            # Apply torque to each joint
            torques['knee'] = peak * self.parameters.get_parameter('knee_assist_weight') * scale
            torques['hip'] = peak * self.parameters.get_parameter('hip_assist_weight') * scale
            torques['ankle'] = peak * self.parameters.get_parameter('ankle_assist_weight') * scale
        
        return torques
    
    def generate_control_trajectory(self, duration=1.0, sample_rate=100):
        """
        Generate a control trajectory for a simulation of specified duration.
        
        Args:
            duration (float): Duration in seconds
            sample_rate (int): Samples per second
            
        Returns:
            dict: Control signals for each actuator over time
        """
        # Assume each gait cycle is 1 second (simplified)
        num_samples = int(duration * sample_rate)
        time_points = np.linspace(0, duration, num_samples)
        
        # Generate control signals for each actuator
        control_signals = {
            'knee': np.zeros(num_samples),
            'hip': np.zeros(num_samples),
            'ankle': np.zeros(num_samples),
            'time': time_points
        }
        
        # Fill in control signals based on gait cycle
        for i, t in enumerate(time_points):
            # Convert time to gait cycle percent (assuming 1s cycle)
            cycle_time = t % 1.0
            gait_percent = cycle_time * 100.0
            
            # Get torques at this point in gait cycle
            torques = self.generate_torque_profile(gait_percent)
            
            # Store in control signals
            for joint in ['knee', 'hip', 'ankle']:
                control_signals[joint][i] = torques[joint]
        
        return control_signals 