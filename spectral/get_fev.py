import numpy as np
import torch
# import copy
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig
import torch.nn.functional as F
from pymatting.util.util import row_sum
from scipy.sparse import diags
import math

def get_diagonal (W):
    D = row_sum(W)
    D[D < 1e-12] = 1.0  # Prevent division by zero.
    D = diags(D)
    return D

def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def get_eigs (feats, modality, how_many = None, device="cpu"):
    # print(device)
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

    if modality == 'text':
        fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device)  ) )

    return torch.abs(fev)
    # return fev

def get_resblock_grad_eigs1(feats, modality, grad, resblock_grad, device="cpu", how_many=None):
    """
    This function computes the gradient eigenvalues, including information from both the attention gradient and resblock gradient.
    
    Args:
        feats: The features to operate on.
        modality: Either 'image' or 'text', depending on the data.
        grad: The gradient from the attention layer.
        resblock_grad: The gradient from the resblocks.
        device: The device to use (default: "cpu").
        how_many: Optionally specify how many eigenvalues to compute.

    Returns:
        Tensor with computed eigenvalues for the gradient.
    """
    # Compute the eigenvectors (fev) as before
    fev = get_eigs(feats, modality, how_many)
    n_feats = fev.size(0)

    # Adjust grad and fev based on the modality
    if n_feats == grad.size(2) - 1:  # images
        grad = grad[:, :, 1:, 1:]
        resblock_grad = resblock_grad[1:]  # Adjust x accordingly
    elif modality == "text":  # text
        grad = grad[:, :, 1:-1, 1:-1]
        fev = fev[1:-1]
        resblock_grad = resblock_grad[1:-1]

    # Process grad
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])  # Shape: [Batch, Seq_len, Seq_len]
    grad = grad.clamp(min=0).mean(dim=0)  # Shape: [Seq_len, Seq_len]
    grad_weight = grad.mean(dim=1)  # Shape: [Seq_len]

    # Process x (gradient from resblock)
    resblock_grad = resblock_grad.squeeze(1)  # Shape: [Seq_len, Hidden_dim]
    x_weight = resblock_grad.clamp(min=0).mean(dim=1)  # Shape: [Seq_len]

    # Combine the weights from grad and x
    combined_weights = grad_weight + x_weight  # Element-wise multiplication: [Seq_len]

    # Apply the combined weights to fev
    fev = fev.to(device)
    weighted_fev = combined_weights * fev  # Shape: [Seq_len]

    # Adjust for text modality by adding zeros at the start and end
    if modality == 'text':
        weighted_fev = torch.cat((torch.zeros(1).to(device), weighted_fev, torch.zeros(1).to(device)))

    return torch.abs(weighted_fev)

def get_resblock_grad_eigs2(feats, modality, grad, x, device="cpu", how_many=None):
    # Compute the eigenvectors (fev) as before
    fev = get_eigs(feats, modality, how_many)
    n_feats = fev.size(0)

    # Adjust grad and fev based on the modality
    if n_feats == grad.size(2) - 1:  # images
        grad = grad[:, :, 1:, 1:]
        x = x[1:]  # Adjust x accordingly
    elif modality == "text":  # text
        grad = grad[:, :, 1:-1, 1:-1]
        fev = fev[1:-1]
        x = x[1:-1]

    # Process grad
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])  # Shape: [Batch, Seq_len, Seq_len]
    grad = grad.clamp(min=0).mean(dim=0)  # Shape: [Seq_len, Seq_len]
    grad_weight = grad.mean(dim=1)  # Shape: [Seq_len]

    # Process x (gradient from resblock)
    x = x.squeeze(1)  # Shape: [Seq_len, Hidden_dim]
    x_weight = x.clamp(min=0).mean(dim=1)  # Shape: [Seq_len]

    # Combine the weights from grad and x
    combined_weights = grad_weight * x_weight  # Element-wise multiplication: [Seq_len]

    # Apply the combined weights to fev
    fev = fev.to(device)
    weighted_fev = combined_weights * fev  # Shape: [Seq_len]

    # Adjust for text modality by adding zeros at the start and end
    if modality == 'text':
        weighted_fev = torch.cat((torch.zeros(1).to(device), weighted_fev, torch.zeros(1).to(device)))

    return torch.abs(weighted_fev)

# def get_resblock_grad_eigs(feats, modality, grad, x, device = "cpu", how_many = None):
#     # print(feats.shape)
#     # print(x.shape)
#     x=  x.squeeze(1)
#     fev = get_eigs(x, modality, how_many)
#     n_feats = fev.size(0)
#     if n_feats == grad.size(2) - 1: #images
#         grad = grad[:, :, 1:, 1:]
#     elif modality == "text": #text
#         grad = grad[:, :, 1:-1, 1:-1]
#         fev = fev[1:-1]

#     grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
#     grad = grad.clamp(min=0).mean(dim=0)
#     fev = fev.to(device)
#     fev = grad @ fev.unsqueeze(1)
#     fev = fev[:, 0]

#     if modality == 'text':
#         fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device)  ) )
    
#     return torch.abs(fev)

# def get_resblock_grad_eigs(feats, modality, grad, x, device = "cpu", how_many = None):
#     # print(feats.shape)
#     # print(x.shape)
#     x=  x.squeeze(1)
#     print(x==feats)
#     # do elementwise multiplication between elements in x and feats
#     x = x * feats
#     # do all elements in x by 2
#     x = x / 2
#     fev = get_eigs(x, modality, how_many)
#     n_feats = fev.size(0)
#     if n_feats == grad.size(2) - 1: #images
#         grad = grad[:, :, 1:, 1:]
#     elif modality == "text": #text
#         grad = grad[:, :, 1:-1, 1:-1]
#         fev = fev[1:-1]

#     grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
#     grad = grad.clamp(min=0).mean(dim=0)
#     fev = fev.to(device)
#     fev = grad @ fev.unsqueeze(1)
#     fev = fev[:, 0]

#     if modality == 'text':
#         fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device)  ) )
    
#     return torch.abs(fev)

# def get_resblock_grad_eigs(feats, modality, grad, x, device = "cpu", how_many = None):
#     # print(feats.shape)
#     # print(x.shape)
#     x=  x.squeeze(1)
#     fev = get_eigs(x, modality, how_many)
#     fev1 = get_eigs(feats, modality, how_many)
#     n_feats = fev.size(0)
#     if n_feats == grad.size(2) - 1: #images
#         grad = grad[:, :, 1:, 1:]
#     elif modality == "text": #text
#         grad = grad[:, :, 1:-1, 1:-1]
#         fev = fev[1:-1]

#     grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
#     grad = grad.clamp(min=0).mean(dim=0)
#     fev = fev.to(device)
#     fev1 = fev1.to(device)
#     fev = grad @ fev.unsqueeze(1) - fev1.unsqueeze(1)
#     fev = fev[:, 0]

#     if modality == 'text':
#         fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device)  ) )
    
#     return torch.abs(fev)

def get_resblock_grad_eigs(feats, modality, grad, x, device="cpu", how_many=None):
    x = x.squeeze(1)
    fev = get_eigs(x, modality, how_many)
    fev1 = get_eigs(feats, modality, how_many)
    n_feats = fev.size(0)
    
    if n_feats == grad.size(2) - 1:  # images
        grad = grad[:, :, 1:, 1:]
    elif modality == "text":  # text
        grad = grad[:, :, 1:-1, 1:-1]
        fev = fev[1:-1]

    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    grad = grad.clamp(min=0).mean(dim=0)
    fev = fev.to(device)
    fev1 = fev1.to(device)

    # Compute x and y
    x_val = grad @ fev.unsqueeze(1)
    y_val = x_val - fev1.unsqueeze(1)

    # Compute z as the element-wise minimum (intersection) of x and y
    z = torch.min(x_val, y_val)

    # Compute a and the final fev
    a =  1- (x_val - z)
    fev = y_val - a
    fev = fev[:, 0]

    # Apply softmax to fev
    fev = torch.softmax(fev, dim=0)

    if modality == 'text':
        fev = torch.cat((torch.zeros(1).to(device), fev, torch.zeros(1).to(device)))

    return torch.abs(fev)


# def get_resblock_grad_eigs(feats, modality, grad, x, device="cpu", how_many=None):
#     x = x.squeeze(1)
#     fev = get_eigs(x, modality, how_many)
#     fev1 = get_eigs(feats, modality, how_many)
#     n_feats = fev.size(0)
    
#     if n_feats == grad.size(2) - 1:  # images
#         grad = grad[:, :, 1:, 1:]
#     elif modality == "text":  # text
#         grad = grad[:, :, 1:-1, 1:-1]
#         fev = fev[1:-1]
#         fev1 = fev1[1:-1]
    
#     grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
#     grad = grad.clamp(min=0).mean(dim=0)
#     fev = fev.to(device)
#     fev1 = fev1.to(device)
    
#     # Compute the absolute difference between fev and fev1
#     fev_diff = torch.abs(fev - fev1)
    
#     # Compute the heatmap by weighting the difference with the gradients
#     heatmap = grad @ fev_diff.unsqueeze(1)
#     heatmap = heatmap[:, 0]
    
#     # Normalize the heatmap
#     heatmap = heatmap / (heatmap.max() + 1e-8)
    
#     if modality == 'text':
#         heatmap = torch.cat((torch.zeros(1).to(device), heatmap, torch.zeros(1).to(device)))
    
#     return heatmap





def get_grad_eigs (feats, modality, grad, device = "cpu", how_many = None):
    fev = get_eigs(feats, modality, how_many)
    n_feats = fev.size(0)
    if n_feats == grad.size(2) - 1: #images
        grad = grad[:, :, 1:, 1:]
    elif modality == "text": #text
        grad = grad[:, :, 1:-1, 1:-1]
        fev = fev[1:-1]

    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    grad = grad.clamp(min=0).mean(dim=0)
    fev = fev.to(device)
    fev = grad @ fev.unsqueeze(1)
    fev = fev[:, 0]

    if modality == 'text':
        fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device)  ) )
    
    return torch.abs(fev)


def get_grad_cam_eigs (feats, modality, grad, cam, device = "cpu", how_many = None):
    fev = get_eigs(feats, modality, how_many)
    n_feats = fev.size(0)
    if n_feats == grad.size(2) - 1: #images
        grad = grad[:, :, 1:, 1:]
    elif modality == "text": #text
        grad = grad[:, :, 1:-1, 1:-1]
        fev = fev[1:-1]
        print("Text grad",grad.shape)

    if n_feats == cam.size(2) - 1: #images
        cam = cam[:, :, 1:, 1:]
    elif modality == "text": #text
        cam = cam[:, :, 1:-1, 1:-1]
        print("Text cam", cam.shape)

    cam = avg_heads(cam, grad)
    print("AVG",cam.shape)
    fev = fev.to(device)
    fev = cam @ fev.unsqueeze(1)
    fev = fev[:, 0]
    if modality == "text":
        fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device) ) )

    return torch.abs(fev)


 

