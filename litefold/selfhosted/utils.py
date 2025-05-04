import numpy as np

def get_ca_coordinates(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca = residue['CA'].get_vector().get_array()
                    coords.append(ca)
    return np.array(coords)

def calculate_distogram(coords):
    dist_matrix = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=-1
    )
    return dist_matrix.tolist()
