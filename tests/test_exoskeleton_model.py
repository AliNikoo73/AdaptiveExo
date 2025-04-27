"""
Tests for the exoskeleton model module.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.opensim_integration.exoskeleton_model import ExoskeletonParameters, ExoskeletonController


class TestExoskeletonParameters(unittest.TestCase):
    """Test cases for ExoskeletonParameters class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exo_params = ExoskeletonParameters()
    
    def test_default_ranges(self):
        """Test that default parameter ranges are properly initialized."""
        self.assertIn('peak_torque', self.exo_params.param_ranges)
        self.assertIn('onset_percent', self.exo_params.param_ranges)
        self.assertIn('knee_assist_weight', self.exo_params.param_ranges)
    
    def test_parameter_setting(self):
        """Test setting parameters within valid ranges."""
        # Set a parameter to a valid value
        result = self.exo_params.set_parameter('peak_torque', 50.0)
        self.assertTrue(result)
        self.assertEqual(self.exo_params.get_parameter('peak_torque'), 50.0)
        
        # Set a parameter to an invalid value (too high)
        result = self.exo_params.set_parameter('peak_torque', 200.0)
        self.assertFalse(result)
        self.assertEqual(self.exo_params.get_parameter('peak_torque'), 50.0)  # Should remain unchanged
        
        # Set a parameter to an invalid value (too low)
        result = self.exo_params.set_parameter('peak_torque', -10.0)
        self.assertFalse(result)
        self.assertEqual(self.exo_params.get_parameter('peak_torque'), 50.0)  # Should remain unchanged
        
        # Set an unknown parameter
        result = self.exo_params.set_parameter('nonexistent_param', 1.0)
        self.assertFalse(result)
    
    def test_random_parameters(self):
        """Test generating random parameters."""
        random_params = self.exo_params.get_random_parameters()
        
        # Check that all parameters are included
        for param in self.exo_params.param_ranges:
            self.assertIn(param, random_params)
            
            # Check that values are within range
            min_val, max_val = self.exo_params.param_ranges[param]
            self.assertGreaterEqual(random_params[param], min_val)
            self.assertLessEqual(random_params[param], max_val)
    
    def test_opensim_parameters(self):
        """Test conversion to OpenSim parameters."""
        # Set test values
        self.exo_params.set_parameter('peak_torque', 80.0)
        self.exo_params.set_parameter('onset_percent', 20.0)
        self.exo_params.set_parameter('offset_percent', 60.0)
        self.exo_params.set_parameter('knee_assist_weight', 0.5)
        
        opensim_params = self.exo_params.get_opensim_parameters()
        
        # Check knee parameters
        self.assertEqual(opensim_params['knee']['joint'], 'knee_angle_r')
        self.assertEqual(opensim_params['knee']['max_torque'], 80.0 * 0.5)  # peak_torque * knee_assist_weight
        self.assertEqual(opensim_params['knee']['onset'], 20.0 / 100.0)  # onset_percent / 100
        self.assertEqual(opensim_params['knee']['offset'], 60.0 / 100.0)  # offset_percent / 100


class TestExoskeletonController(unittest.TestCase):
    """Test cases for ExoskeletonController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exo_params = ExoskeletonParameters()
        self.exo_params.set_parameter('peak_torque', 50.0)
        self.exo_params.set_parameter('onset_percent', 10.0)
        self.exo_params.set_parameter('offset_percent', 60.0)
        self.exo_params.set_parameter('knee_assist_weight', 1.0)
        self.exo_params.set_parameter('hip_assist_weight', 0.5)
        self.exo_params.set_parameter('ankle_assist_weight', 0.0)
        
        self.controller = ExoskeletonController(parameters=self.exo_params)
    
    def test_torque_profile(self):
        """Test torque profile generation."""
        # Test before onset - should be zero
        torques = self.controller.generate_torque_profile(5.0)
        self.assertEqual(torques['knee'], 0.0)
        self.assertEqual(torques['hip'], 0.0)
        self.assertEqual(torques['ankle'], 0.0)
        
        # Test at middle of active phase - should be maximum
        torques = self.controller.generate_torque_profile(35.0)
        self.assertEqual(torques['knee'], 50.0)  # peak_torque * knee_weight
        self.assertEqual(torques['hip'], 25.0)   # peak_torque * hip_weight
        self.assertEqual(torques['ankle'], 0.0)  # peak_torque * ankle_weight (0)
        
        # Test after offset - should be zero
        torques = self.controller.generate_torque_profile(65.0)
        self.assertEqual(torques['knee'], 0.0)
        self.assertEqual(torques['hip'], 0.0)
        self.assertEqual(torques['ankle'], 0.0)
    
    def test_control_trajectory(self):
        """Test control trajectory generation."""
        # Generate trajectory
        controls = self.controller.generate_control_trajectory(duration=1.0, sample_rate=100)
        
        # Check dimensions
        self.assertEqual(len(controls['time']), 100)
        self.assertEqual(len(controls['knee']), 100)
        self.assertEqual(len(controls['hip']), 100)
        self.assertEqual(len(controls['ankle']), 100)
        
        # Check time points
        self.assertEqual(controls['time'][0], 0.0)
        self.assertAlmostEqual(controls['time'][-1], 0.99, places=2)  # Should be close to 1.0


if __name__ == '__main__':
    unittest.main() 