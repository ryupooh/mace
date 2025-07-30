import importlib.util
from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list

try:
    import torch
    from torch_cluster import radius_graph
except ImportError:
    torch = None

def _check_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None

def _get_neighborhood_matscipy(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    # temp_cell = np.copy(cell)
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )


    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge
 
        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]
 
    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]
 
    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell 

def _get_neighborhood_torch(
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


if  _check_package_available("torch") and _check_package_available("torch_cluster"):
    _is_neighborhood_torch_available = True
    print("get_neighborhood will use torch_cluster for neighborhood detection. It'll be applied only if no PBC is used.")


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if _is_neighborhood_torch_available and not any(pbc):
        return _get_neighborhood_torch(
            positions, cutoff, pbc=pbc, cell=cell, true_self_interaction=true_self_interaction
        )
    else:
       return _get_neighborhood_matscipy(
            positions, cutoff, pbc=pbc, cell=cell, true_self_interaction=true_self_interaction
        )