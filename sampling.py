import numpy as np
import torch

device = 'cuda'
truncated = 0.8
n_classes = 120
nz = 120


def sample(model, zs, batch_size=32, class_id=None, aux_labels=None):
    out = []

    n_images = zs.shape[0]
    n_batches = int(np.ceil(n_images / batch_size))

    if aux_labels is None:
        aux_labels = np.full(n_images, class_id)

    for i_batch in range(n_batches):
        batch_idx_start = i_batch*batch_size
        batch_idx_end = min((i_batch+1)*batch_size, n_images)

        gen_z = zs[batch_idx_start:batch_idx_end]

        batch_aux_labels = aux_labels[batch_idx_start:batch_idx_end]
        aux_labels_ohe = np.eye(n_classes)[batch_aux_labels]
        aux_labels_ohe = torch.from_numpy(aux_labels_ohe[:, :])
        aux_labels_ohe = aux_labels_ohe.float().to(device)

        with torch.no_grad():
            gen_images = model(gen_z, aux_labels_ohe)
        gen_images = gen_images.to('cpu')

        # denormalize
        gen_images = gen_images * 0.5 + 0.5
        out.append(gen_images)

    return torch.cat(out, dim=0)


def linear_interpolation(start_z, end_z, steps):
    interpolation_steps = torch.linspace(0.0, 1.0, steps, device=device)
    z = start_z[None] + (end_z[None, :] - start_z[None, :]) * interpolation_steps[:, None]
    return z


def get_cycling_grid_same(height, width, cycles, frames_per_half_cycle):
    all_zs = torch.empty((height, width, cycles * 2 * frames_per_half_cycle, 120), device=device)

    z_shared = torch.rand(120, device=device) * truncated
    for c in range(cycles):
        z_individual = torch.rand(120, device=device) * truncated
        z_individual = z_individual.unsqueeze(0).unsqueeze(0).expand(height, width, -1)

        f_idx_start = c * 2 * frames_per_half_cycle
        f_idx_end = f_idx_start + frames_per_half_cycle
        for h in range(height):
            for w in range(width):
                all_zs[h, w, f_idx_start:f_idx_end] = linear_interpolation(z_shared, z_individual[h, w], frames_per_half_cycle)

        z_shared = torch.rand(120, device=device) * truncated

        f_idx_start = c * 2 * frames_per_half_cycle + frames_per_half_cycle
        f_idx_end = f_idx_start + frames_per_half_cycle
        for h in range(height):
            for w in range(width):
                all_zs[h, w, f_idx_start:f_idx_end] = linear_interpolation(z_individual[h, w], z_shared, frames_per_half_cycle)

    return all_zs


def get_cycling_grid(height, width, cycles, frames_per_half_cycle):
    all_zs = torch.empty((height, width, cycles * 2 * frames_per_half_cycle, 120), device=device)

    z_shared = torch.rand(120, device=device) * truncated
    for c in range(cycles):
        z_individual = torch.rand((height, width, 120), device=device) * truncated
        f_idx_start = c * 2 * frames_per_half_cycle
        f_idx_end = f_idx_start + frames_per_half_cycle
        for h in range(height):
            for w in range(width):
                all_zs[h, w, f_idx_start:f_idx_end] = linear_interpolation(z_shared, z_individual[h, w], frames_per_half_cycle)

        z_shared = torch.rand(120, device=device) * truncated

        f_idx_start = c * 2 * frames_per_half_cycle + frames_per_half_cycle
        f_idx_end = f_idx_start + frames_per_half_cycle
        for h in range(height):
            for w in range(width):
                all_zs[h, w, f_idx_start:f_idx_end] = linear_interpolation(z_individual[h, w], z_shared, frames_per_half_cycle)

    return all_zs
