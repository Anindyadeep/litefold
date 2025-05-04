import numpy as np
from io import StringIO
from prody import parsePDB, writePDB, calcTransformation, AtomGroup
import tempfile
import os
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa

def get_ca_coordinates(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca = residue['CA'].get_vector().get_array()
                    coords.append(ca)
    return np.array(coords)


def calculate_distogram(coords, num_bins=36, max_distance=20):
    # Calculate pairwise distances
    dists = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    
    # Instead of returning a full 3D tensor, return a more compact representation
    # - dists: the actual distance matrix
    # - bins: the bin edges for interpretation
    bin_edges = np.linspace(0, max_distance, num_bins + 1)
    
    # Clip distances to max_distance
    dists = np.minimum(dists, max_distance)
    
    # Return a dict with the distance matrix and bin information
    return {
        "distance_matrix": dists.tolist(),
        "bin_edges": bin_edges.tolist(),
        "max_distance": max_distance,
        "num_bins": num_bins
    }


def extract_sequence_from_pdb_content(pdb_content):
    """Extracts the amino acid sequence from PDB file content
    
    Args:
        pdb_content: String containing PDB file content
        
    Returns:
        String containing the one-letter amino acid sequence
    """
    # Create a temporary file to parse the PDB content
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as temp_file:
        temp_file.write(pdb_content)
        temp_file.flush()
        
        try:
            # Parse the PDB file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", temp_file.name)
            
            # Extract the sequence
            sequence = ""
            # Use the first model
            model = structure[0]
            
            # Go through all chains and compile the sequence
            for chain in model:
                chain_sequence = ""
                for residue in chain:
                    # Skip non-amino acid residues (water, ligands, etc.)
                    if is_aa(residue):
                        try:
                            # Convert three-letter code to one-letter code
                            resname = residue.get_resname()
                            one_letter = three_to_one(resname)
                            chain_sequence += one_letter
                        except KeyError:
                            # If residue is not standard, use X
                            chain_sequence += 'X'
                
                # Add separator between chains if there are multiple
                if chain_sequence:
                    if sequence:
                        sequence += "|"  # Separate chains with |
                    sequence += chain_sequence
            
            return sequence
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)


def align_structures_and_calculate_tmscore(
    predicted_pdb_content: str,
    ground_truth_pdb_content: str,
) -> dict:
    # Create temporary files for the PDB content
    pred_temp = None
    gt_temp = None
    aligned_temp = None
    
    try:
        # Write predicted structure to temp file
        pred_temp = tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False)
        pred_temp.write(predicted_pdb_content)
        pred_temp.close()
        
        # Write ground truth structure to temp file
        gt_temp = tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False)
        gt_temp.write(ground_truth_pdb_content)
        gt_temp.close()
        
        # Parse the PDB files
        predicted = parsePDB(pred_temp.name)
        ground_truth = parsePDB(gt_temp.name)

        predicted_ca = predicted.select('name CA')
        ground_truth_ca = ground_truth.select('name CA')

        if predicted_ca is None or ground_truth_ca is None:
            raise ValueError("No C-alpha atoms found in one of the structures.")
            
        pred_atoms = {
            (atom.getChid(), atom.getResnum()): atom for atom in predicted_ca
        }
        gt_atoms = {
            (atom.getChid(), atom.getResnum()): atom for atom in ground_truth_ca
        }
        common_keys = sorted(set(pred_atoms.keys()) & set(gt_atoms.keys()))
        if not common_keys:
            raise ValueError("No common residues with C-alpha atoms found between structures.")
        
        pred_matched = [pred_atoms[key] for key in common_keys]
        gt_matched = [gt_atoms[key] for key in common_keys]

        # calculate the transformation matrix
        pred_group = AtomGroup('predicted_matched')
        gt_group = AtomGroup('ground_truth_matched')
        pred_group.setCoords([atom.getCoords() for atom in pred_matched])
        gt_group.setCoords([atom.getCoords() for atom in gt_matched])

        # Perform the alignment
        transformation = calcTransformation(pred_group, gt_group)
        transformation.apply(predicted)
            
        # Calculate RMSD between CA atoms
        ca_rmsd = np.sqrt(np.mean(np.sum(
            (np.array([atom.getCoords() for atom in pred_matched]) - 
             np.array([atom.getCoords() for atom in gt_matched]))**2, 
            axis=1)))
        
        # Calculate TM-score
        # L is the length of the target protein
        L = len(ground_truth_ca)
        
        # d0 is a normalization factor
        d0 = 1.24 * (L - 15)**(1/3) - 1.8
        
        # Calculate distances between aligned pairs
        distances = np.sqrt(np.sum(
            (np.array([atom.getCoords() for atom in pred_matched]) - 
             np.array([atom.getCoords() for atom in gt_matched]))**2, 
            axis=1))
        
        # TM-score formula
        tm_score = np.sum(1 / (1 + (distances / d0)**2)) / L
        
        # Write the aligned structure to a temporary file then read it
        aligned_temp = tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False)
        writePDB(aligned_temp.name, predicted)
        aligned_temp.close()
        
        # Read the aligned PDB content
        with open(aligned_temp.name, 'r') as f:
            aligned_pdb = f.read()
        
        return {
            "tm_score": float(tm_score),
            "rmsd": float(ca_rmsd),
            "aligned_pdb_content": aligned_pdb
        }
    
    finally:
        # Clean up temporary files
        if pred_temp and os.path.exists(pred_temp.name):
            os.unlink(pred_temp.name)
        if gt_temp and os.path.exists(gt_temp.name):
            os.unlink(gt_temp.name)
        if aligned_temp and os.path.exists(aligned_temp.name):
            os.unlink(aligned_temp.name)
 