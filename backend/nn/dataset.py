import torch
import numpy as np
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, x_path, y_path, seq_length=50, augment=False):
        self.augment = augment
        self.seq_length = seq_length
        self.X, self.y = self._process_data(x_path, y_path)

    def _process_data(self, x_path, y_path):
        try:
            X = np.load(x_path)
            y = np.load(y_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data files not found at {x_path} or {y_path}")

        X_seq, y_seq = [], []
        stride = 20
        
        # Convert flat frames to sequences
        for i in range(0, len(X) - self.seq_length, stride):
            clip = X[i : i+self.seq_length]
            labels = y[i : i+self.seq_length]
            
            # Robust Labeling: Majority vote with bias for critical classes
            crit_count = np.sum(labels == 2)
            bad_count = np.sum(labels == 1)
            threshold = self.seq_length * 0.2
            
            if crit_count > threshold: label = 2
            elif bad_count > threshold: label = 1
            else: label = 0
            
            # Reshape: (50, 102) -> (50, 17, 6)
            clip = clip.reshape(self.seq_length, 17, 6)
            X_seq.append(clip)
            y_seq.append(label)
            
        return np.array(X_seq), torch.tensor(np.array(y_seq), dtype=torch.long)

    def _augment_physics(self, clip):
        """Apply random rotation and jitter."""
        # Rotation around gravity (Y-axis)
        theta = np.random.uniform(-0.3, 0.3) 
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        # Separate Pos and Vel
        pos = clip[:, :, 0:3].reshape(-1, 3)
        vel = clip[:, :, 3:6].reshape(-1, 3)
        
        # Rotate both
        pos = np.dot(pos, rot_mat.T).reshape(self.seq_length, 17, 3)
        vel = np.dot(vel, rot_mat.T).reshape(self.seq_length, 17, 3)
        
        # Add Jitter
        pos += np.random.normal(0, 0.002, pos.shape)

        if np.random.rand() < 0.2:
            # Set x,y,z, vx,vy,vz to 0 for legs
            clip[:, 11:17, :] = 0 
            
        # 10% chance to zero out one arm (Simulate side view blockage)
        if np.random.rand() < 0.1:
            if np.random.rand() < 0.5:
                clip[:, 5:11:2, :] = 0 # Left Arm
            else:
                clip[:, 6:12:2, :] = 0 # Right Arm
                
        return np.concatenate((pos, vel), axis=2)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        clip = self.X[idx]
        if self.augment:
            clip = self._augment_physics(clip)
        
        # To Tensor: (Time, Vertices, Channels) -> (Channels, Time, Vertices)
        clip_tensor = torch.tensor(clip, dtype=torch.float32).permute(2, 0, 1)
        return clip_tensor, self.y[idx]