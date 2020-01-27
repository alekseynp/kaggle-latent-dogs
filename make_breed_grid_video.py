import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator
from sampling import sample, get_cycling_grid_same, n_classes

device = 'cuda'

model = Generator(n_feat=36, codes_dim=24, n_classes=n_classes).to(device)
model.load_state_dict(torch.load('generator.pth'))

height = 8
width = 15
cycles = 8
frames_per_half_cycle = 24*4
all_zs = get_cycling_grid_same(height, width, cycles, frames_per_half_cycle)

for f_idx in tqdm(range(all_zs.size(2))):
    z_batch = all_zs[:, :, f_idx]
    z_batch = z_batch.view(-1, 120)

    aux_labels = np.arange(120)

    images = sample(model, z_batch, aux_labels=aux_labels)

    frame = torch.empty((3, height * 64, width * 64), device=device)
    for h in range(height):
        for w in range(width):
            frame[:, h * 64:(h + 1) * 64, w * 64:(w + 1) * 64] = images[h * width + w]
    save_image(frame, 'frames/{}.png'.format(f_idx))
