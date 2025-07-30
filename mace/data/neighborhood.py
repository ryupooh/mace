from typing import Optional, Tuple

import numpy as np
import torch
from torch_cluster import radius_graph


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = torch.tensor(positions, dtype=torch.float32)

    # radius_graph expects shape [N, 3]
    # It returns edges [2, num_edges], where each column is [sender, receiver]
    edge_index = radius_graph(
        pos,
        r=cutoff,
        loop=true_self_interaction,  # include self-edges if requested
        max_num_neighbors=1000,      # adjust if your system is very dense
    )

    sender = edge_index[0].numpy()
    receiver = edge_index[1].numpy()

    # In torch_cluster, no direct periodic support: you must "tile" the box yourself for PBC.
    # If you need periodic boundary conditions, you must augment the positions array by replicating in neighboring images.
    # For now, let's assume non-periodic (or system is already appropriately wrapped).

    # Compute shifts: positions[receiver] - positions[sender]
    shifts = positions[receiver] - positions[sender]

    # For compatibility, unit_shifts and cell (for now, zeros and given cell)
    unit_shifts = np.zeros_like(shifts)
    # If you want to support PBC, you need to apply minimum image convention here.

    # Optionally, filter out self-edges if not wanted and true_self_interaction is False
    if not true_self_interaction:
        mask = sender != receiver
        sender = sender[mask]
        receiver = receiver[mask]
        shifts = shifts[mask]
        unit_shifts = unit_shifts[mask]

    edge_index = np.stack((sender, receiver))
    return edge_index, shifts, unit_shifts, cell