from ultralytics import YOLO
import os
from pathlib import Path

def train_weapon_model():
    # Path to the base model
    model_path = 'yolov8m.pt'
    
    # Check if model exists, if not it will download automatically
    model = YOLO(model_path)
    
    # Path to the dataset config
    data_path = 'weapon_data.yaml'
    
    print(f"Starting training on {data_path} using {model_path}...")
    
    # Train the model
    # We use a relatively small number of epochs for demonstration
    # In a real scenario, you'd want 50-100 epochs
    results = model.train(
        data=data_path,
        epochs=2,
        imgsz=640,
        batch=8,
        name='weapon_accuracy_fix',
        exist_ok=True,
        resume=True
    )
    
    print("Training complete!")
    print(f"Results saved to: {results.save_dir}")
    
    # Path to the best weights
    best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_weights.exists():
        print(f"Best model available at: {best_weights}")

if __name__ == "__main__":
    train_weapon_model()
