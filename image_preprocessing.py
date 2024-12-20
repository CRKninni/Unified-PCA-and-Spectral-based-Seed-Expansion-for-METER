import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from meter.transforms import clip_transform

# Define your transformation
IMG_SIZE = 576
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def preprocess_and_save_image(image_path, processed_dir):
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Processed image path with .pt extension
    processed_image_path = os.path.join(processed_dir, os.path.splitext(os.path.basename(image_path))[0] + ".pt")

    # Check if the processed image already exists
    if not os.path.exists(processed_image_path):
        # Load and transform the image
        image = Image.open(image_path)
        processed_image = clip_transform(size=IMG_SIZE)(image)
        processed_image = processed_image.unsqueeze(0)  # Add batch dimension
        
        # Save the processed tensor as a .pt file
        torch.save(processed_image, processed_image_path)
    else:
        print(f"Processed image already exists: {processed_image_path}")

# Example usage
COCO_path = "/home/pranav/ExplanableAI/CoCo/val2014/"
processed_dir = "/home/pranav/ExplanableAI/CoCo/val2014_processed/"
for img_name in os.listdir(COCO_path):
    img_path = os.path.join(COCO_path, img_name)
    preprocess_and_save_image(img_path, processed_dir)
