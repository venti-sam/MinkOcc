import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 1. CONFIGURATION ---

# The base directory where your data is located.
BASE_DIR = "." 

# Names of the input and output directories
IMAGE_DIR_NAME = "CAM_FRONT"
NPZ_PARENT_DIR_NAME = "CAM_FRONT_SAM/2d"
OUTPUT_DIR_NAME = "unknown_class_visualizations"

# Define the set of class IDs you already know.
KNOWN_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17}

# The color to use for the overlay. Since we are only showing one mask
# at a time, we can use the same bright color for all of them.
# RGBA = (Red, Green, Blue, Alpha)
OVERLAY_COLOR = [255, 0, 0, 153] # Bright Red with ~60% opacity

# --- 2. SCRIPT LOGIC ---

def visualize_individual_unknowns():
    # Construct full paths
    image_dir = os.path.join(BASE_DIR, IMAGE_DIR_NAME)
    npz_dir = os.path.join(BASE_DIR, NPZ_PARENT_DIR_NAME)
    output_dir = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)

    # --- Setup ---
    print("Starting individual mask visualization process...")
    if not os.path.isdir(image_dir):
        print(f"ERROR: Image directory not found at '{image_dir}'")
        return
    if not os.path.isdir(npz_dir):
        print(f"ERROR: NPZ directory not found at '{npz_dir}'")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output images will be saved in '{output_dir}'")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    print(f"Found {len(image_files)} images to process.")

    # --- Main Loop (Iterating through each image) ---
    for image_filename in image_files:
        base_name, _ = os.path.splitext(image_filename)
        npz_filename = base_name + ".npz"
        
        image_path = os.path.join(image_dir, image_filename)
        npz_path = os.path.join(npz_dir, npz_filename)

        if not os.path.exists(npz_path):
            continue

        try:
            # --- Load Data for the Pair ---
            raw_image = Image.open(image_path).convert("RGBA")
            with np.load(npz_path) as data:
                label_map = data['label_map.npy']

            # --- Identify Unknown Classes ---
            present_ids = np.unique(label_map)
            unknown_ids = sorted([pid for pid in present_ids if pid not in KNOWN_CLASSES])

            if not unknown_ids:
                continue
            
            print(f"Processing {image_filename}: Found unknown classes {unknown_ids}")

            # --- Inner Loop (Iterating through each UNKNOWN class) ---
            for class_id in unknown_ids:
                # Create a boolean mask for ONLY the current unknown class
                binary_mask = (label_map == class_id)
                
                # Create the overlay image
                overlay_image = Image.new('RGBA', raw_image.size, (0, 0, 0, 0))
                overlay_pixels = np.array(overlay_image)
                overlay_pixels[binary_mask] = OVERLAY_COLOR
                overlay_image = Image.fromarray(overlay_pixels)
                
                # Composite the single mask onto the original image
                final_image = Image.alpha_composite(raw_image, overlay_image)

                # --- Add Text Label ---
                draw = ImageDraw.Draw(final_image)
                try:
                    font = ImageFont.truetype("arial.ttf", size=30)
                except IOError:
                    font = ImageFont.load_default()
                
                text = f"Visualizing Class ID: {class_id}"
                bbox = draw.textbbox((10, 10), text, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 150))
                draw.text((10, 10), text, fill=(255, 255, 255), font=font)

                # --- Save the Final Image with a descriptive name ---
                output_filename = f"{base_name}_class_{class_id}.png"
                output_path = os.path.join(output_dir, output_filename)
                final_image.save(output_path)
                print(f"  -> Saved visualization for class {class_id} to '{output_path}'")

        except Exception as e:
            print(f"!! ERROR processing {image_filename}: {e}")

    print("\nBatch processing complete.")

# --- Run the script ---
if __name__ == "__main__":
    visualize_individual_unknowns()