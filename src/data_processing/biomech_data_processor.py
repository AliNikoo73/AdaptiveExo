"""
Processing module for biomechanical data.
This module handles data preprocessing, feature extraction, and normalization.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib.pyplot as plt


class BiomechanicalDataProcessor:
    """
    Class for processing and extracting features from biomechanical data.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data processor.
        
        Args:
            data_dir (str, optional): Directory containing raw data files
        """
        self.data_dir = data_dir
        self.processed_data = None
        self.feature_sets = {}
    
    def load_motion_data(self, file_path=None, subject_id=None):
        """
        Load motion data from file.
        
        Args:
            file_path (str, optional): Path to data file
            subject_id (str, optional): Subject ID if loading from data_dir
            
        Returns:
            DataFrame: Loaded motion data
        """
        if file_path is None and subject_id is not None and self.data_dir:
            file_path = os.path.join(self.data_dir, f"subject_{subject_id}.csv")
        
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        try:
            # Load data based on file extension
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.sto'):
                data = self._load_opensim_storage(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            return data
        
        except Exception as e:
            print(f"Error loading motion data: {e}")
            return None
    
    def _load_opensim_storage(self, file_path):
        """
        Load data from OpenSim .sto file format.
        
        Args:
            file_path (str): Path to .sto file
            
        Returns:
            DataFrame: Loaded data
        """
        # Simple .sto parser - in practice would use osim.Storage
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header row
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('endheader'):
                header_idx = i + 1
                break
        
        # Get column names
        col_names = lines[header_idx].strip().split()
        
        # Parse data rows
        data_rows = []
        for line in lines[header_idx+1:]:
            if line.strip():
                values = [float(x) for x in line.strip().split()]
                data_rows.append(values)
        
        # Create DataFrame
        return pd.DataFrame(data_rows, columns=col_names)
    
    def preprocess_data(self, data, sampling_rate=100, lowpass_cutoff=10, normalize=True):
        """
        Preprocess the raw motion data.
        
        Args:
            data (DataFrame): Raw motion data
            sampling_rate (int): Data sampling rate in Hz
            lowpass_cutoff (float): Cutoff frequency for lowpass filter in Hz
            normalize (bool): Whether to normalize the data
            
        Returns:
            DataFrame: Preprocessed data
        """
        # Copy data to avoid modifying original
        processed = data.copy()
        
        # Remove any rows with NaN values
        processed = processed.dropna()
        
        # Filter columns that contain motion data (exclude time, etc.)
        motion_cols = [col for col in processed.columns if col not in ['time', 'subject_id', 'trial']]
        
        # Apply low-pass filter to motion data columns
        if lowpass_cutoff:
            nyquist = 0.5 * sampling_rate
            normal_cutoff = lowpass_cutoff / nyquist
            b, a = signal.butter(4, normal_cutoff, btype='low')
            
            for col in motion_cols:
                processed[col] = signal.filtfilt(b, a, processed[col])
        
        # Normalize data if requested
        if normalize:
            for col in motion_cols:
                mean_val = processed[col].mean()
                std_val = processed[col].std()
                if std_val > 0:
                    processed[col] = (processed[col] - mean_val) / std_val
        
        self.processed_data = processed
        return processed
    
    def segment_gait_cycles(self, data=None, method='heel_strike', time_col='time'):
        """
        Segment data into individual gait cycles.
        
        Args:
            data (DataFrame, optional): Motion data (uses self.processed_data if None)
            method (str): Method for identifying gait cycles
            time_col (str): Name of the time column
            
        Returns:
            list: List of DataFrames, each containing one gait cycle
        """
        if data is None:
            if self.processed_data is None:
                raise ValueError("No data available for segmentation")
            data = self.processed_data
        
        cycles = []
        
        if method == 'heel_strike':
            # Look for heel strikes in right heel marker (z-coordinate)
            # This is a simplified approach - real implementation would be more robust
            if 'heel_marker_r_z' in data.columns:
                # Find local minima in heel marker height
                heel_z = data['heel_marker_r_z'].values
                minima_indices = signal.find_peaks(-heel_z)[0]
                
                # Extract cycles between consecutive heel strikes
                for i in range(len(minima_indices) - 1):
                    start_idx = minima_indices[i]
                    end_idx = minima_indices[i + 1]
                    cycle = data.iloc[start_idx:end_idx].copy()
                    
                    # Add percent through gait cycle
                    cycle_time = cycle[time_col] - cycle[time_col].iloc[0]
                    cycle_duration = cycle_time.iloc[-1]
                    cycle['percent_gait'] = (cycle_time / cycle_duration) * 100.0
                    
                    cycles.append(cycle)
            else:
                # Fallback: simply divide into 1-second segments (assuming 1 Hz gait cycle)
                cycle_duration = 1.0  # seconds
                cycle_samples = int(cycle_duration * 100)  # assuming 100 Hz
                
                for i in range(0, len(data) - cycle_samples, cycle_samples):
                    cycle = data.iloc[i:i+cycle_samples].copy()
                    cycle['percent_gait'] = np.linspace(0, 100, len(cycle))
                    cycles.append(cycle)
        
        return cycles
    
    def extract_features(self, data=None, feature_set='basic'):
        """
        Extract features from motion data for ML model input.
        
        Args:
            data (DataFrame, optional): Motion data (uses self.processed_data if None)
            feature_set (str): Name of feature set to extract
            
        Returns:
            DataFrame: Extracted features
        """
        if data is None:
            if self.processed_data is None:
                raise ValueError("No data available for feature extraction")
            data = self.processed_data
        
        # Dictionary to store extracted features
        features = {}
        
        if feature_set == 'basic':
            # Extract basic statistical features from each column
            for col in data.columns:
                if col not in ['time', 'subject_id', 'trial', 'percent_gait']:
                    features[f"{col}_mean"] = data[col].mean()
                    features[f"{col}_std"] = data[col].std()
                    features[f"{col}_min"] = data[col].min()
                    features[f"{col}_max"] = data[col].max()
                    features[f"{col}_range"] = data[col].max() - data[col].min()
        
        elif feature_set == 'biomechanical':
            # Extract biomechanical-specific features
            # Joint angles at key gait cycle points
            if 'percent_gait' in data.columns:
                gait_points = [0, 25, 50, 75]
                for point in gait_points:
                    closest_idx = (data['percent_gait'] - point).abs().idxmin()
                    for col in data.columns:
                        if '_angle_' in col and col not in ['time', 'subject_id', 'trial', 'percent_gait']:
                            features[f"{col}_at_{point}"] = data.loc[closest_idx, col]
            
            # ROM (range of motion) for each joint
            for col in data.columns:
                if '_angle_' in col and col not in ['time', 'subject_id', 'trial', 'percent_gait']:
                    features[f"{col}_rom"] = data[col].max() - data[col].min()
            
            # Symmetry features (compare left vs right)
            for col in data.columns:
                if '_r_' in col and col.replace('_r_', '_l_') in data.columns:
                    r_col = col
                    l_col = col.replace('_r_', '_l_')
                    features[f"{r_col.split('_r_')[0]}_asymmetry"] = data[r_col].mean() - data[l_col].mean()
                    features[f"{r_col.split('_r_')[0]}_rom_asymmetry"] = (data[r_col].max() - data[r_col].min()) - (data[l_col].max() - data[l_col].min())
        
        elif feature_set == 'temporal':
            # Temporal features from time series
            for col in data.columns:
                if col not in ['time', 'subject_id', 'trial', 'percent_gait']:
                    # ZCR (Zero Crossing Rate)
                    zcr = np.sum(np.diff(np.signbit(data[col].values))) / len(data[col])
                    features[f"{col}_zcr"] = zcr
                    
                    # FFT features - frequency domain
                    if len(data[col]) > 10:  # Need enough samples
                        fft_vals = np.abs(np.fft.rfft(data[col].values))
                        fft_freq = np.fft.rfftfreq(len(data[col].values))
                        
                        # Dominant frequency
                        dom_freq_idx = np.argmax(fft_vals)
                        features[f"{col}_dom_freq"] = fft_freq[dom_freq_idx]
                        
                        # Spectral centroid
                        if np.sum(fft_vals) > 0:
                            spectral_centroid = np.sum(fft_freq * fft_vals) / np.sum(fft_vals)
                            features[f"{col}_spec_centroid"] = spectral_centroid
        
        # Store the extracted features
        self.feature_sets[feature_set] = pd.DataFrame([features])
        return self.feature_sets[feature_set]
    
    def extract_features_from_cycles(self, cycles, feature_set='basic'):
        """
        Extract features from a list of gait cycles.
        
        Args:
            cycles (list): List of DataFrames containing gait cycles
            feature_set (str): Name of feature set to extract
            
        Returns:
            DataFrame: Features for each gait cycle
        """
        all_features = []
        
        for i, cycle in enumerate(cycles):
            features = self.extract_features(cycle, feature_set)
            features['cycle_id'] = i
            all_features.append(features)
        
        # Combine all features
        combined = pd.concat(all_features).reset_index(drop=True)
        return combined
    
    def combine_subject_features(self, subject_ids, data_dir=None, feature_set='basic'):
        """
        Load, process, and extract features for multiple subjects.
        
        Args:
            subject_ids (list): List of subject IDs
            data_dir (str, optional): Directory containing data files
            feature_set (str): Feature set to extract
            
        Returns:
            DataFrame: Combined features for all subjects
        """
        if data_dir:
            self.data_dir = data_dir
        
        all_features = []
        
        for subject_id in subject_ids:
            try:
                data = self.load_motion_data(subject_id=subject_id)
                processed = self.preprocess_data(data)
                cycles = self.segment_gait_cycles(processed)
                features = self.extract_features_from_cycles(cycles, feature_set)
                features['subject_id'] = subject_id
                all_features.append(features)
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
        
        if all_features:
            return pd.concat(all_features).reset_index(drop=True)
        return pd.DataFrame()
    
    def visualize_data(self, data=None, columns=None, title="Motion Data Visualization"):
        """
        Visualize the motion data.
        
        Args:
            data (DataFrame, optional): Data to visualize
            columns (list, optional): Columns to include in visualization
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Generated figure
        """
        if data is None:
            if self.processed_data is None:
                raise ValueError("No data available for visualization")
            data = self.processed_data
        
        if columns is None:
            # Select motion data columns (exclude metadata)
            columns = [col for col in data.columns if col not in ['time', 'subject_id', 'trial', 'percent_gait']]
            # Limit to first 6 columns for readability
            columns = columns[:6]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot time series data
        x_col = 'percent_gait' if 'percent_gait' in data.columns else 'time'
        for col in columns:
            ax.plot(data[x_col], data[col], label=col)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def save_processed_data(self, output_dir, filename=None, subject_id=None):
        """
        Save processed data to file.
        
        Args:
            output_dir (str): Directory to save output
            filename (str, optional): Output filename
            subject_id (str, optional): Subject ID
            
        Returns:
            str: Path to saved file
        """
        if self.processed_data is None:
            raise ValueError("No processed data available to save")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not specified
        if filename is None:
            if subject_id:
                filename = f"processed_subject_{subject_id}.csv"
            else:
                filename = "processed_data.csv"
        
        # Save to CSV
        output_path = os.path.join(output_dir, filename)
        self.processed_data.to_csv(output_path, index=False)
        
        return output_path 