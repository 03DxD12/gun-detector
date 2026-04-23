import pandas as pd
import os
from pathlib import Path

# Paths
metadata_path = r"C:\Users\Denmhar\.cache\kagglehub\datasets\snehilsanyal\weapon-detection-test\versions\5\metadata.csv"
dataset_dir = r"C:\Users\Denmhar\.cache\kagglehub\datasets\snehilsanyal\weapon-detection-test\versions\5\weapon_detection"

# Load metadata
df = pd.read_csv(metadata_path)

def fix_labels(subset):
    labels_dir = Path(dataset_dir) / subset / "labels"
    print(f"Fixing labels in {labels_dir}...")
    
    for index, row in df.iterrows():
        label_file = row['labelfile']
        target_class = row['target']
        
        file_path = labels_dir / label_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    parts[0] = str(target_class)
                    new_lines.append(" ".join(parts) + "\n")
            
            with open(file_path, 'w') as f:
                f.writelines(new_lines)

fix_labels("train")
fix_labels("val")
print("Dataset re-labeling complete!")
