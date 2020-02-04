import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator
from sampling import sample_truncated_normal, linear_interpolation, n_classes

device = 'cuda'
truncated = 0.8

model = Generator(n_feat=36, codes_dim=24, n_classes=n_classes).to(device)
model.load_state_dict(torch.load('generator.pth'))

z_points = [sample_truncated_normal(120, truncated, device) for _ in range(16)]
zs = []
for i in range(5 - 1):
    zs.append(linear_interpolation(z_points[i], z_points[i+1], 24 * 4))
zs = torch.cat(zs, 0).float()


for j in tqdm(range(zs.size()[0])):
    z = zs[j].unsqueeze(0)

    aux_labels = np.full(1, 0)
    aux_labels_ohe = np.eye(n_classes)[aux_labels]
    aux_labels_ohe = torch.from_numpy(aux_labels_ohe[:, :]).float().cuda()

    out = model(z, aux_labels_ohe, get_feature_maps=True)

    frame = []

    for i in range(len(out) - 1):
        x = out[i][0]
        mins = x.min(1, keepdim=True)[0].min(2, keepdim=True)[0]
        maxs = x.max(1, keepdim=True)[0].max(2, keepdim=True)[0]
        x = (x - mins) / (maxs - mins)

        resolution = x.size()[-1]
        panels = x.size()[0]
        height = 512 // resolution
        width = int(np.ceil(panels / height))

        frame_part = torch.empty((3, 512, width * resolution), device=device)
        for p in range(panels):
            frame_part[:, ((p % height)*resolution):((p % height) + 1)*resolution, (p // height)*resolution:((p // height) + 1)*resolution] = x[p]
        frame.append(frame_part)

    [f.size() for f in frame]

    widths = [f.size()[-1] for f in frame]
    full_width = sum(widths) + 512

    full_frame = torch.empty((3, 512, full_width), device=device)

    w_offset = 0
    for f in frame:
        _, h, w = f.size()
        padding = (512 - h) // 2
        full_frame[:, padding:(padding + h), w_offset:(w_offset + w)] = f
        w_offset += w

    full_frame[:, :, (full_width - 512):] = torch.nn.functional.interpolate(out[-1] * 0.5 + 0.5, 512, mode='nearest')[0]

    save_image(full_frame, 'viz/{}.png'.format(j))

