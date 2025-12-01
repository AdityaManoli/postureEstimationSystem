import numpy as np

def align_to_camera(skeleton):
    """
    Rotates the skeleton around the Y-axis (gravity) so the hips face forward.
    This makes the model View-Invariant (robust to diagonal/side views).
    """
    # COCO Indices: 11=LeftHip, 12=RightHip
    left_hip = skeleton[11]
    right_hip = skeleton[12]
    
    # 1. Calculate the Hip Vector (Left -> Right)
    # We want this vector to be parallel to the Global X-Axis (1, 0, 0)
    hip_vec = right_hip - left_hip
    
    # 2. Calculate Angle in the X-Z plane (Top-down view)
    # We ignore Y because we rotate around gravity
    dx, dz = hip_vec[0], hip_vec[2]
    theta = np.arctan2(dz, dx)
    
    # We want hips to point along X-axis, so we rotate by -theta
    # However, standard T-pose has hips along X.
    rotation_angle = -theta
    
    # 3. Create Rotation Matrix (around Y-axis)
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])
    
    # 4. Apply Rotation to all points
    # skeleton shape: (17, 3). Transpose for matmul -> (3, 17)
    aligned_skeleton = np.dot(rotation_matrix, skeleton.T).T
    
    return aligned_skeleton

def normalize_skeleton(skeleton_17):
    """
    Standardizes a 17-point skeleton to be Scale-Invariant AND View-Invariant.
    Input: (17, 3) [x, y, z]
    Output: (17, 3) Normalized
    """
    # 1. CENTER THE BODY AT HIPS (0,0,0) -> Position Invariant
    left_hip = skeleton_17[11]
    right_hip = skeleton_17[12]
    hip_center = (left_hip + right_hip) / 2.0
    centered = skeleton_17 - hip_center

    # 2. ALIGN TO CAMERA (The New Fix) -> View/Perspective Invariant
    aligned = align_to_camera(centered)

    # 3. SCALE BY TORSO LENGTH -> Size/Distance Invariant
    # Recalculate indices on the aligned body
    left_shoulder = aligned[5]
    right_shoulder = aligned[6]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    
    spine_len = np.linalg.norm(shoulder_center) # Distance from (0,0,0)
    
    if spine_len < 1e-6:
        return aligned # Return unscaled if invalid
        
    return aligned / spine_len