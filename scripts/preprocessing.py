import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

df_loaded = pd.read_pickle('filtered_data_distance_less_25.pkl')

preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_chunk(args):
    """Verwerk een lijst van (idx, row)-tuples."""
    chunk, output_dir, transform = args
    for idx, row in chunk:
        img_path = row['image_path']
        label = row['label']
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            output_path = os.path.join(output_dir, f"{idx}.pt")
            torch.save({'image': image, 'label': label}, output_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def preprocess_and_save_parallel(dataframe, output_dir, transform, num_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    dataframe = dataframe.dropna(subset=['bouwjaar', 'image_path']).reset_index(drop=True)

    # Label aanmaken
    min_year = 1900
    max_year = 2024
    bin_width = 20
    dataframe['label'] = ((dataframe['bouwjaar'] - min_year) // bin_width).clip(0, ((max_year - min_year) // bin_width))

    # Verdeel de DataFrame in chunks als LISTS (geen generators!)
    num_workers = num_workers or cpu_count()
    indices_rows = list(dataframe.iterrows())  # Converteer naar lijst van (idx, row)-tuples
    chunk_size = len(indices_rows) // num_workers
    chunks = [
        indices_rows[i * chunk_size : (i + 1) * chunk_size] 
        for i in range(num_workers)
    ]

    # Start parallelle verwerking
    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap(
                process_chunk, 
                [(chunk, output_dir, transform) for chunk in chunks]
            ),
            total=len(chunks),
            desc="Parallel preprocessing"
        ))

# Zorg ervoor dat je dit in een __main__-blok zet op Windows!
if __name__ == '__main__':
    # Laad je dataframe (df_loaded) hier
    preprocess_and_save_parallel(df_loaded, "preprocessed_v2", preprocess_transform, num_workers=4)