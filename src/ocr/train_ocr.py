import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import OCRDataset
from trocr_model import load_model

# -----------------------------
# PATHS
# -----------------------------
image_dir = "data/raw/word_level/train/images"
label_file = "data/raw/word_level/train/labels.txt"

print("🔹 Starting OCR training script...")

# -----------------------------
# DEVICE
# -----------------------------
print("🔹 Checking device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# -----------------------------
# LOAD MODEL
# -----------------------------
print("\n🔹 Loading TrOCR model and processor...")
processor, model = load_model()
print("✅ Model loaded successfully")

print("🔹 Model config values:")
print("   pad_token_id:", model.config.pad_token_id)
print("   decoder_start_token_id:", model.config.decoder_start_token_id)
print("   eos_token_id:", model.config.eos_token_id)

model.to(device)
model.train()
print("✅ Model moved to device and set to train mode")

# -----------------------------
# LOAD DATASET
# -----------------------------
print("\n🔹 Loading dataset...")
dataset = OCRDataset(image_dir, label_file, processor)
print(f"✅ Dataset loaded successfully with {len(dataset)} samples")

print("\n🔹 Testing first sample from dataset...")
sample = dataset[0]
print("✅ First sample loaded successfully")
print("   pixel_values shape:", sample["pixel_values"].shape)
print("   labels shape:", sample["labels"].shape)

# -----------------------------
# DATALOADER
# -----------------------------
print("\n🔹 Creating DataLoader...")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(f"✅ DataLoader created with {len(dataloader)} batches")

# -----------------------------
# OPTIMIZER
# -----------------------------
print("\n🔹 Initializing optimizer...")
optimizer = AdamW(model.parameters(), lr=5e-5)
print("✅ Optimizer initialized")

# -----------------------------
# TRAINING LOOP
# -----------------------------
EPOCHS = 3
print(f"\n🚀 Starting training for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    print(f"\n========== EPOCH {epoch+1}/{EPOCHS} ==========")
    total_loss = 0

    for step, batch in enumerate(dataloader):
        print(f"\n➡️ Batch {step+1}/{len(dataloader)}")

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        print("   Forward pass...")
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        print(f"   Loss: {loss.item():.4f}")

        print("   Backward pass...")
        loss.backward()

        print("   Optimizer step...")
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"\n✅ Epoch {epoch+1} completed")
    print(f"📉 Total Loss: {total_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
print("\n💾 Saving fine-tuned model...")
model.save_pretrained("models/trocr_finetuned")
processor.save_pretrained("models/trocr_finetuned")

print("✅ Model saved successfully to models/trocr_finetuned")
print("\n🎉 Training completed!")