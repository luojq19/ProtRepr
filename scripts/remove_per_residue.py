import sys
sys.path.append('.')
import torch
import os
from tqdm import tqdm

data_dir = 'data/sprot_esm1b_emb_per_residue'
output_dir = 'data/sprot_esm1b_emb'
for root, dirs, files in os.walk(data_dir):
    print(f'num of emb files: {len(files)}')
    for file in tqdm(files):
        if not file.endswith('.pt'):
            continue
        emb = torch.load(os.path.join(root, file))
        # remove key 'representations' from emb
        emb.pop('representations')
        torch.save(emb, os.path.join(output_dir, file))