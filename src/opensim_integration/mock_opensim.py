"""
Mock OpenSim module for demonstration purposes.
This module provides mock classes to simulate OpenSim functionality.
"""

class MockModel:
    """Mock OpenSim Model class"""
    def __init__(self, model_path=None):
        self.name = "Mock OpenSim Model"
        self.coordinates = {
            'knee_angle_r': 0.0,
            'hip_flexion_r': 0.0,
            'ankle_angle_r': 0.0,
            'knee_angle_l': 0.0,
            'hip_flexion_l': 0.0,
            'ankle_angle_l': 0.0
        }
        self.forces = []
    
    def initSystem(self):
        """Initialize the model system"""
        return MockState()
    
    def getCoordinateSet(self):
        """Get the coordinate set"""
        return MockCoordinateSet(self.coordinates)
    
    def addForce(self, force):
        """Add a force to the model"""
        self.forces.append(force)
        return True
    
    def finalizeConnections(self):
        """Finalize model connections"""
        return True
    
    def setUseVisualizer(self, value):
        """Set visualizer usage"""
        self.use_visualizer = value


class MockState:
    """Mock OpenSim State class"""
    def __init__(self):
        self.time = 0.0
        self.values = {}


class MockCoordinateSet:
    """Mock OpenSim CoordinateSet class"""
    def __init__(self, coordinates):
        self.coordinates = coordinates
    
    def get(self, name):
        if name in self.coordinates:
            return MockCoordinate(name)
        return None


class MockCoordinate:
    """Mock OpenSim Coordinate class"""
    def __init__(self, name):
        self.name = name
        self.value = 0.0
        self.speed = 0.0
    
    def setValue(self, state, value):
        """Set coordinate value"""
        self.value = value
        state.values[self.name] = value
    
    def setSpeedValue(self, state, value):
        """Set coordinate speed value"""
        self.speed = value
        state.values[f"{self.name}_speed"] = value


class MockCoordinateActuator:
    """Mock OpenSim CoordinateActuator class"""
    def __init__(self):
        self.name = ""
        self.coordinate = None
        self.optimal_force = 0.0
        self.min_control = 0.0
        self.max_control = 0.0
    
    def setName(self, name):
        """Set actuator name"""
        self.name = name
    
    def setCoordinate(self, coordinate):
        """Set actuator coordinate"""
        self.coordinate = coordinate
    
    def setOptimalForce(self, force):
        """Set optimal force"""
        self.optimal_force = force
    
    def setMinControl(self, value):
        """Set minimum control value"""
        self.min_control = value
    
    def setMaxControl(self, value):
        """Set maximum control value"""
        self.max_control = value


class MockManager:
    """Mock OpenSim Manager class"""
    def __init__(self, model):
        self.model = model
        self.initial_time = 0.0
        self.final_time = 0.0
        self.accuracy = 0.0
        self.state = MockState()
        self.storage = MockStorage()
    
    def setInitialTime(self, time):
        """Set initial time"""
        self.initial_time = time
    
    def setFinalTime(self, time):
        """Set final time"""
        self.final_time = time
    
    def setIntegratorAccuracy(self, accuracy):
        """Set integrator accuracy"""
        self.accuracy = accuracy
    
    def initialize(self, state):
        """Initialize manager"""
        self.state = state
    
    def integrate(self, duration):
        """Run integration"""
        self.state.time = self.initial_time + duration
        return self.state
    
    def getStateStorage(self):
        """Get state storage"""
        return self.storage


class MockStorage:
    """Mock OpenSim Storage class"""
    def __init__(self, file_path=None):
        self.data = {}
        self.times = [0.0, 0.5, 1.0]
        
        # Initialize with some mock data if file path provided
        if file_path:
            self._init_mock_data()
    
    def _init_mock_data(self):
        """Initialize with mock data"""
        self.data['knee_angle_r'] = [0.0, 20.0, 0.0]
        self.data['hip_flexion_r'] = [0.0, 30.0, 0.0]
        self.data['ankle_angle_r'] = [0.0, 10.0, 0.0]
    
    def print(self, file_path):
        """Save storage to file"""
        return True
    
    def getLastTime(self):
        """Get last time in storage"""
        return self.times[-1] if self.times else 0.0
    
    def getFirstTime(self):
        """Get first time in storage"""
        return self.times[0] if self.times else 0.0
    
    def getSize(self):
        """Get storage size"""
        return len(self.times) 