# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

# MERU-specific imports
from meru.config import LazyConfig, LazyFactory
from meru.utils.checkpointing import CheckpointManager

# --- Configuration ---
# IMPORTANT: Update these paths to point to your files.
TRAIN_CONFIG_PATH = './configs/train_meru_vit_b.py'  # Path to the training configuration file (.py)
CHECKPOINT_PATH = './checkpoints/meru_vit_b.pth'    # Path to the model checkpoint file (.pth)
IMAGE_FOLDER_PATH = './datasets/fashion200k/casual_and_day_dresses'  # Path to the root folder containing image subdirectories
OUTPUT_JSON_PATH = './embeddings/hyperbolic_embeddings.json' # Path to save the final JSON file

def main():
    """
    Main function to load the model, process images, and save embeddings.
    """
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model Loading ---
    print("Loading model from checkpoint...")
    # Create the model using the training configuration and load the pre-trained weights.
    _C_TRAIN = LazyConfig.load(TRAIN_CONFIG_PATH)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(CHECKPOINT_PATH)
    print("Model loaded successfully.")

    # --- Image Preprocessing ---
    # Define the same image transformations used during training for consistency.
    # This pipeline resizes, center-crops, and converts images to tensors.
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    # --- Image File Discovery ---
    # Collect all relevant image file paths from the specified folder.
    # It looks for images ending with _0.jpeg, _0.jpg, or _0.png in the subdirectories.
    image_paths = []
    for subfolder in os.listdir(IMAGE_FOLDER_PATH):
        subfolder_path = os.path.join(IMAGE_FOLDER_PATH, subfolder)
        if os.path.isdir(subfolder_path):
            for image_file in os.listdir(subfolder_path):
                if image_file.lower().endswith(('_0.jpeg', '_0.jpg', '_0.png')):
                    image_paths.append(os.path.join(subfolder_path, image_file))
    
    print(f"Found {len(image_paths)} images to process.")

    # --- Embedding Generation ---
    # Dictionary to store the generated embeddings.
    image_embeddings = {}

    # Process each image with a progress bar (tqdm).
    # torch.no_grad() is used as we are in inference mode and don't need to calculate gradients.
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Generating Embeddings"):
            try:
                # Open image, ensure it's in RGB format, and apply preprocessing.
                image = Image.open(image_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                # Generate the image embedding using the model's encoder.
                # The 'project=True' argument is used to get the final projected features.
                embedding = model.encode_image(image_input, project=True)
                
                # L2-normalize the embedding vector.
                embedding /= embedding.norm(dim=-1, keepdim=True)
                
                # Use the image's unique ID (part of the filename before '_') as the key.
                file_name_id = os.path.basename(image_path).split('_')[0]
                
                # Store the embedding in the dictionary.
                # We move it to the CPU and convert it to a standard Python list for JSON serialization.
                image_embeddings[file_name_id] = embedding.cpu().numpy().tolist()

            except Exception as e:
                print(f"\nCould not process image {image_path}. Error: {e}")

    # --- Save Embeddings to JSON ---
    # Create the output directory if it doesn't exist.
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # Save the dictionary of embeddings as a JSON file.
    with open(OUTPUT_JSON_PATH, 'w') as json_file:
        json.dump(image_embeddings, json_file, indent=4)

    print(f"\nProcessing complete. Embeddings saved to '{OUTPUT_JSON_PATH}'")

if __name__ == "__main__":
    main()
