import torch
from object_discovery import *

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig
from pymatting.util.util import row_sum
from scipy.sparse import diags
from sklearn.decomposition import PCA
import math




def get_grad_cam(grads,cams,modality):
    final_gradcam = []
    num_layers = len(cams)
    for i in range(num_layers):
        # Get gradients and attention maps for this layer
        grad = grads[i]  # Shape: (1, 12, 1297, 1297)
        cam = cams[i]    # Shape: (1, 12, 1297, 1297)
        
        if modality == "image":
            # Remove the [CLS] token (i.e., the first token) and keep only the 1296 image tokens
            cam = cam[:, :, 1:, 1:]  # Shape: (1, 12, 1297, 1296)
            grad = grad[:, :, 1:, 1:].clamp(0)  # Shape: (1, 12, 1297, 1296)
        elif modality == "text":
            cam = cam[:,:,1:-1,1:-1]
            grad = grad[:,:,1:-1,1:-1].clamp(0)
        else:
            print("Invalid modality")
            return None

        # Multiply gradients by attention maps (element-wise)
        layer_gradcam = cam * grad  # Still (1, 12, 1297, 1296)

        # Average over the 12 heads to get a single map per layer
        layer_gradcam = layer_gradcam.mean(1)  # Shape: (1, 1297, 1296)

        final_gradcam.append(layer_gradcam.cpu())
        
        
    if modality == "image":
        final_gradcam_temp = np.mean(final_gradcam, axis=0)
        final_gradcam_temp = final_gradcam_temp.squeeze()
        
        gradcam = final_gradcam_temp  # Shape: (1296, 36, 36)
        heatmap = np.mean(gradcam, axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.flatten() , final_gradcam
        
        
    elif modality == "text":
        final_gradcam_temp = np.mean(final_gradcam, axis=0)
        final_gradcam_temp = final_gradcam_temp.squeeze(0)
        text_relearance = np.mean(final_gradcam_temp, axis=0)
        text_relearance = (text_relearance - text_relearance.min()) / (text_relearance.max() - text_relearance.min() + 1e-8)
        
        return text_relearance , final_gradcam


def get_rollout(cams, modality):
    num_layers = len(cams)
    x = None

    for i in range(num_layers):
        
        if modality == "image":
            cam_i = cams[i][0]
        elif modality == "text":
            cam_i = cams[i][0][:,1:-1,1:-1]
        else:
            print("Invalid modality")
            return None
            
        cam_i_avg = cam_i.mean(dim=0) 
        
        if x is None:
            x = cam_i_avg.clone()
        else:
            x = x * cam_i_avg  # Element-wise multiplication
            x = x / torch.norm(x, p=2)
            
    if modality == "image":
        final_gradcam_temp = x[:1296, :1296].reshape(1296, 36, 36)
        final_gradcam_temp = final_gradcam_temp.cpu().detach().numpy()
        gradcam = final_gradcam_temp
    elif modality == "text":
        try:
            gradcam = x.cpu().numpy()
        except:
            gradcam = x.detach().cpu().numpy()

    heatmap = np.mean(gradcam, axis=0)
    # normalise heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap.flatten()

def get_diagonal (W):
    D = row_sum(W)
    D[D < 1e-12] = 1.0  # Prevent division by zero.
    D = diags(D)
    return D

def get_eigs (feats, modality, how_many = None, device="cpu"):
    if feats.size(0) == 1:
        feats = feats.detach().squeeze()


    if modality == "image":
        n_image_feats = feats.size(0)
        val = int( math.sqrt(n_image_feats) )
        if val * val == n_image_feats:
            feats = F.normalize(feats, p = 2, dim = -1).to(device)
        elif val * val + 1 == n_image_feats:
            feats = F.normalize(feats, p = 2, dim = -1)[1:].to(device)
        else:
            print(f"Invalid number of features detected: {n_image_feats}")

    else:
        feats = F.normalize(feats, p = 2, dim = -1)[1:-1].to(device)

    W_feat = (feats @ feats.T)
    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max() 

    W_feat = W_feat.detach().cpu().numpy()

    
    D = np.array(get_diagonal(W_feat).todense())

    L = D - W_feat

    L_shape = L.shape[0]
    if how_many >= L_shape - 1: 
        how_many = L_shape - 2

    try:
        eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5, M = D)
    except:
        try:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5)
        except:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM')
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
    
    n_tuple = torch.kthvalue(eigenvalues.real, 2)
    fev_idx = n_tuple.indices
    fev = eigenvectors[fev_idx].to(device)
        
    fev = torch.abs(fev)
    fevs_final = (fev - fev.min()) / (fev.max() - fev.min() + 1e-8)

    return fevs_final

def eig_seed(feats, modality, iters)        :
    patch_scores_norm = get_eigs(feats, modality, how_many = 5)
    num_patches = int(np.sqrt(len(patch_scores_norm)))
    heatmap = patch_scores_norm.reshape(num_patches, num_patches)  # Shape: [num_patches, num_patches]

    
    seed_index = np.argmax(patch_scores_norm)

    # Convert the 1D index to 2D indices
    seed_row = seed_index // num_patches
    seed_col = seed_index % num_patches


    # Initialize a mask for the expanded seed region
    seed_mask = np.zeros_like(heatmap)
    seed_mask[seed_row, seed_col] = 1

    # Define the number of expansion iterations
    num_expansion_iters = iters

    # Perform seed expansion
    for _ in range(num_expansion_iters):
        # Find neighboring patches
        neighbor_mask = cv2.dilate(seed_mask, np.ones((3, 3), np.uint8), iterations=1)
        neighbor_mask = neighbor_mask - seed_mask  # Exclude already included patches
        neighbor_indices = np.where(neighbor_mask > 0)
        
        # For each neighbor, decide whether to include it based on similarity
        for r, c in zip(*neighbor_indices):
            # Use heatmap values as similarity scores
            similarity = heatmap[r, c]
            # Define a threshold for inclusion
            threshold = 0.5  # Adjust this value as needed
            
            if similarity >= threshold:
                seed_mask[r, c] = 1  # Include the neighbor
            else:
                seed_mask[r, c] = 0.001

    # Apply the seed mask to the heatmap
    refined_heatmap = heatmap * seed_mask
    
    return refined_heatmap.flatten()
    
def get_pca_component(feats, modality, component=0, device="cpu"):
    if feats.size(0) == 1:
        feats = feats.detach().squeeze()

    if modality == "image":
        n_image_feats = feats.size(0)
        val = int(math.sqrt(n_image_feats))
        if val * val == n_image_feats:
            feats = F.normalize(feats, p=2, dim=-1).to(device)
        elif val * val + 1 == n_image_feats:
            feats = F.normalize(feats, p=2, dim=-1)[1:].to(device)
        else:
            print(f"Invalid number of features detected: {n_image_feats}")
    else:
        feats = F.normalize(feats, p=2, dim=-1)[1:-1].to(device)

    # Reshape feats to apply PCA on (1296, 768) as desired
    feats_reshaped = feats.cpu().detach().numpy()

    # Apply PCA on the reshaped data to get the second principal component
    pca = PCA(n_components=5)
    principal_components = pca.fit_transform(feats_reshaped)

    # Extract the second principal component and expand to original shape
    second_pc = principal_components[:, component]

    # Convert to tensor and move to the specified device
    second_pc = torch.tensor(second_pc, dtype=torch.float32).to(device)


    second_pc = torch.abs(second_pc)
    
    # Normalize the second principal component for visualization
    second_pc_norm = (second_pc - second_pc.min()) / (second_pc.max() - second_pc.min() + 1e-8)
    
    return second_pc_norm

def get_image_relevance(ret, grads, cams):
    dsm = get_eigs(ret['image_feats'], "image", how_many = 5)
    lost = eig_seed(ret['image_feats'], "image", 15)
    pca_0 = get_pca_component(ret['image_feats'], "image", 0)

    dsm = np.array(dsm)
    lost = np.array(lost)
    pca_0 = np.array(pca_0)
    
    x = np.array([dsm, lost, pca_0])
    x = np.sum(x,axis=0)
    
    grad_cam, a = get_grad_cam(grads, cams,"image")
    grad_cam = grad_cam * 2
    rollout = get_rollout(cams,"image")
    pca_1 = get_pca_component(ret['image_feats'], "image", 1)
    pca_1 = np.array(pca_1) * 0.001
    
    y = np.array([grad_cam,rollout, pca_1])
    
    y = np.sum(y,axis=0)
    
    z = x + y
    
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
    return z

def get_text_relevance(ret, grads, cam):
    pca_0 = get_pca_component(ret['text_feats'], "text", 0)
    pca_0 = np.array(pca_0)
    
    x = np.array([pca_0])
    x = np.sum(x,axis=0)
    
    grad_cam, a = get_grad_cam(grads, cam, "text")
    rollout = get_rollout(cam, "text")
    pca_1 = get_pca_component(ret['text_feats'], "text", 1)
    pca_1 = np.array(pca_1) * 0.01
    
    y = np.array([grad_cam,rollout, pca_1])
    y = np.sum(y,axis=0)
    
    z = x + y
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
    return z