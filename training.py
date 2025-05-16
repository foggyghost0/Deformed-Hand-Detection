# pip install torch torchvision transformers datasets ultralytics hand-yolov5

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import (
    ViTHybridFeatureExtractor,
    ViTHybridForImageClassification,
    TrainingArguments,
    Trainer,
)


class HandCropDataset(Dataset):
    def __init__(self, root_dir: str, feature_extractor, crop_ratio=0.75):
        """
        root_dir:
          - class_0/
          - class_1/
        crop_ratio: fraction of final crop occupied by detected hand bbox
        """
        self.paths = []
        for label, cls in enumerate(sorted(os.listdir(root_dir))):
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                self.paths.append((os.path.join(cls_dir, img), label))

        # Load a YOLOv5 model fine-tuned for hand detection
        # (hand-yolov5 repo uses same API as ultralytics/yolov5)
        from yolov5 import load  # installed by hand-yolov5

        self.hand_model = load("hand-yolov5")  # or path to custom weights

        self.feature_extractor = feature_extractor
        self.crop_ratio = crop_ratio

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path, label = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Run hand detector
        results = self.hand_model(img)
        detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]
        if len(detections) == 0:
            # fallback to center crop if no hand detected
            crop_box = (0, 0, w, h)
        else:
            # Take the highest-confidence detection
            x1, y1, x2, y2 = detections[0][:4]
            bw, bh = x2 - x1, y2 - y1
            # Compute square crop so bbox occupies crop_ratio of side
            side = max(bw, bh) / self.crop_ratio
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            left = max(0, cx - side / 2)
            top = max(0, cy - side / 2)
            right = min(w, cx + side / 2)
            bottom = min(h, cy + side / 2)
            crop_box = (left, top, right, bottom)

        # Crop & Resize to ViT input (224)
        img_cropped = img.crop(crop_box).resize((224, 224), Image.BICUBIC)

        # Preprocess for ViT-Hybrid
        encoding = self.feature_extractor(images=img_cropped, return_tensors="pt")
        # squeeze batch dim
        pixel_values = encoding.pixel_values.squeeze()
        # Normalize to mean=0.5, std=0.5
        pixel_values = (pixel_values - 0.5) / 0.5

        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}


def main():
    # 3.1. Feature Extractor & Model
    feature_extractor = ViTHybridFeatureExtractor.from_pretrained(
        "google/vit-hybrid-base-bit-384"
    )
    model = ViTHybridForImageClassification.from_pretrained(
        "google/vit-hybrid-base-bit-384",
        num_labels=2,
        id2label={0: "class_0", 1: "class_1"},
        label2id={"class_0": 0, "class_1": 1},
    )

    # 3.2. Dataset & Dataloader
    train_ds = HandCropDataset("data/train", feature_extractor)
    val_ds = HandCropDataset("data/val", feature_extractor)

    # 3.3. TrainingArguments
    training_args = TrainingArguments(
        output_dir="./vit_hybrid_finetuned",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # 3.4. Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    # 3.5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 3.6. Train!
    trainer.train()


if __name__ == "__main__":
    main()
