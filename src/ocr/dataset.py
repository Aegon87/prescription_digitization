import os
from PIL import Image
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, image_dir, label_file, processor):
        self.image_dir = image_dir
        self.processor = processor

        self.samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)

                img_name = parts[0]
                text = parts[1] if len(parts) > 1 else ""

                self.samples.append((img_name, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]

        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        ).input_ids.squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }