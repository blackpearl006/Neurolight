import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np
import random
import datetime
import torch

def plot_roc_curve(all_labels, all_predicted):
        fpr, tpr, _ = roc_curve(all_labels, all_predicted)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.show()
        print("ROC curve saved as roc_curve.png")


def plot_curves(train_losses,val_losses):
    plt.figure(figsize=(4,4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.legend()
    plt.show()

def plot_brain_age_delta(true,predicted,save=True,name='some_name'):
    max_val=int(max(max(true),max(predicted)))
    min_val=int(min(min(true),min(predicted)))
    plt.plot(np.linspace(min_val, max_val), np.linspace(min_val, max_val), color='red', linestyle='--', label='y=x')
    plt.scatter(true,predicted)
    plt.ylim(min_val,max_val)
    plt.xlim(min_val,max_val)
    plt.ylabel('Predicted')
    plt.xlabel('True')
    if save: plt.savefig(f'{name}.png',dpi=400)
    else: plt.show()


def visualize_random_instance(batch):
    # Unpack the batch
    (original, masked, mask), labels = batch

    # Convert tensors to numpy arrays
    original_np = original.numpy()
    masked_np = masked.numpy()
    mask_np = mask.numpy()

    # Select a random index from the batch
    idx = random.randint(0, original_np.shape[0] - 1)
    
    # Select images for this index
    original_image = original_np[idx]
    masked_image = masked_np[idx]
    mask_image = mask_np[idx]

    # Plot the slices
    plt.figure(figsize=(12, 4))
    
    # Original slice
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original_image[0][45], cmap='gray')
    plt.axis('off')
    
    # Masked slice
    plt.subplot(1, 3, 2)
    plt.title("Masked")
    plt.imshow(masked_image[0][45], cmap='gray')
    plt.axis('off')
    
    # Mask overlay
    plt.subplot(1, 3, 3)
    plt.title("Mask Overlay")
    masked_slice = np.where(mask_image[0][45] > 0, original_image[0][45], np.nan)  # Overlay mask on the original
    plt.imshow(masked_slice, cmap='gray')
    plt.imshow(mask_image[0][45], cmap='jet', alpha=0.5)  # Overlay mask in color
    plt.colorbar(label='Mask')
    plt.axis('off')

    plt.show()

def compare_reconstruction(original, masked, mask, labels, reconstructed, centres, filename = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M:%S")):
    center_d, center_h, center_w = centres

    original_np = original.cpu().numpy()
    masked_np = masked.detach().numpy()
    mask_np = mask.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()

    # Select a random index from the batch
    # idx = random.randint(0, original_np.shape[0] - 1)
    idx = 0
    
    # Select images for this index
    original_image = original_np[idx]
    masked_image = masked_np[idx]
    mask_image = mask_np[idx]
    reconstructed_image = reconstructed_np[idx]

    # Plot the slices
    plt.figure(figsize=(9, 9))
    
    # Saggital 
    plt.subplot(3, 3, 1)
    plt.title("Original")
    plt.imshow(np.rot90(original_image[0][center_d[idx],:,:]), cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.title("Reconstructed")
    plt.imshow(np.rot90(reconstructed_image[0][center_d[idx],:,:]), cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.title("Masked")
    plt.imshow(np.rot90(masked_image[0][center_d[idx],:,:]), cmap='gray')
    plt.axis('off')
    
    # Coronal
    plt.subplot(3, 3, 4)
    plt.title("Original")
    plt.imshow(np.rot90(original_image[0][:,center_h[idx],:]), cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.title("Reconstructed")
    plt.imshow(np.rot90(reconstructed_image[0][:,center_h[idx],:]), cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.title("Masked")
    plt.imshow(np.rot90(masked_image[0][:,center_h[idx],:]), cmap='gray')
    plt.axis('off')
    
    # Axial
    plt.subplot(3, 3, 7)
    plt.title("Original")
    plt.imshow(np.rot90(original_image[0][:,:,center_w[idx]]), cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.title("Reconstructed")
    plt.imshow(np.rot90(reconstructed_image[0][:,:,center_w[idx]]), cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.title("Masked")
    plt.imshow(np.rot90(masked_image[0][:,:,center_w[idx]]), cmap='gray')
    plt.axis('off')
    

    plt.suptitle(f'Mask centre at D: {center_d[idx].item()}, H:{center_h[idx].item()}, W:{center_w[idx].item()}')
    plt.tight_layout()
    plt.show()

def compare_mri_slices(subject1, subject2, locations=((45, 50, 55), (60, 65, 70), (45, 50, 55)), rotation_angle=0):
    """
    Compares the sagittal, coronal, and axial slices of two MRI image tensors at specified locations with a given rotation.

    Args:
        subject1 (torch.Tensor): The first MRI image tensor with shape (1, D, H, W).
        subject2 (torch.Tensor): The second MRI image tensor with shape (1, D, H, W).
        locations (tuple): A tuple containing three tuples, each representing the slice locations for (sagittal, coronal, axial).
                           Example: ((D1, D2, D3), (H1, H2, H3), (W1, W2, W3))
        rotation_angle (int): The angle to rotate the images (in degrees). Must be a multiple of 90.

    Example usage:
        # compare_mri_slices(original[0].cpu(), reconstructed[0].cpu(), locations=((45, 50, 55), (60, 65, 70), (45, 50, 55)), rotation_angle=90)
    """
    # Convert rotation angle to equivalent number of 90-degree rotations
    num_rotations = (rotation_angle // 90) % 4
    # Extract the slice locations
    D_locs, H_locs, W_locs = locations

    plt.figure(figsize=(12, 8))

    for i, (D, H, W) in enumerate(zip(D_locs, H_locs, W_locs)):
        # Sagittal view (YZ plane) for both subjects
        sagittal_1 = subject1[0, D, :, :]
        sagittal_2 = subject2[0, D, :, :]
        if num_rotations > 0:
            sagittal_1 = torch.rot90(sagittal_1, num_rotations, [0, 1])
            sagittal_2 = torch.rot90(sagittal_2, num_rotations, [0, 1])
        
        plt.subplot(3, 6, i*6 + 1)
        plt.imshow(sagittal_1, cmap='gray')
        plt.title(f"Original Sagittal {D}")

        plt.subplot(3, 6, i*6 + 2)
        plt.imshow(sagittal_2, cmap='gray')
        plt.title(f"Reconstructed Sagittal {D}")

        # Coronal view (XZ plane) for both subjects
        coronal_1 = subject1[0, :, H, :]
        coronal_2 = subject2[0, :, H, :]
        if num_rotations > 0:
            coronal_1 = torch.rot90(coronal_1, num_rotations, [0, 1])
            coronal_2 = torch.rot90(coronal_2, num_rotations, [0, 1])
        
        plt.subplot(3, 6, i*6 + 3)
        plt.imshow(coronal_1, cmap='gray')
        plt.title(f"Original Coronal {H}")

        plt.subplot(3, 6, i*6 + 4)
        plt.imshow(coronal_2, cmap='gray')
        plt.title(f"Reconstructed Coronal {H}")

        # Axial view (XY plane) for both subjects
        axial_1 = subject1[0, :, :, W]
        axial_2 = subject2[0, :, :, W]
        if num_rotations > 0:
            axial_1 = torch.rot90(axial_1, num_rotations, [0, 1])
            axial_2 = torch.rot90(axial_2, num_rotations, [0, 1])
        
        plt.subplot(3, 6, i*6 + 5)
        plt.imshow(axial_1, cmap='gray')
        plt.title(f"Original Axial {W}")

        plt.subplot(3, 6, i*6 + 6)
        plt.imshow(axial_2, cmap='gray')
        plt.title(f"Reconstructed Axial {W}")

    plt.tight_layout()
    # plt.axis('off')
    plt.show()