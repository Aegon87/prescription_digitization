import os
import shutil

ROOT = r"C:\Users\Anshuman Pandey\OneDrive\Desktop\Prescription_digitalization\data\raw\prescriptions"  
IMAGES_DIR = os.path.join(ROOT, "images")
LABELS_DIR = os.path.join(ROOT, "labels")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

for item in os.listdir(ROOT):
    item_path = os.path.join(ROOT, item)

    # Only process numbered folders
    if os.path.isdir(item_path) and item.isdigit():
        jpg_path = os.path.join(item_path, f"{item}.jpg")
        json_path = os.path.join(item_path, f"{item}.json")

        if os.path.exists(jpg_path):
            shutil.move(jpg_path, os.path.join(IMAGES_DIR, f"{item}.jpg"))
        else:
            print(f"⚠️ Missing image: {jpg_path}")

        if os.path.exists(json_path):
            shutil.move(json_path, os.path.join(LABELS_DIR, f"{item}.json"))
        else:
            print(f"⚠️ Missing label: {json_path}")

print("✅ Flattening complete.")
