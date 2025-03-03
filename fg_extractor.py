from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def extract_functional_groups(mol_list):
    """
    Extract functional groups from a list of RDKit molecules.

    Parameters:
    -----------
    mol_list : list
        List of RDKit molecule objects

    Returns:
    --------
    list of tuples
        Each tuple contains (mol, dictionary) where the dictionary maps
        functional group names to lists of atom indices present in that group
    """
    # Comprehensive dictionary of SMARTS patterns for functional groups
    functional_groups = {
        # Oxygen-containing groups
        'Alcohol': '[OX2H]',
        'Phenol': '[OX2H][cX3]:[cX3]',
        'Primary Alcohol': '[OX2H][CX4H2]',
        'Secondary Alcohol': '[OX2H][CX4H1]',
        'Tertiary Alcohol': '[OX2H][CX4H0]',
        'Aldehyde': '[CX3H1](=O)[#6]',
        'Ketone': '[#6][CX3](=O)[#6]',
        'Carboxylic Acid': '[CX3](=O)[OX2H1]',
        'Ester': '[#6][CX3](=O)[OX2][#6]',
        'Acid Anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
        'Acyl Halide': '[CX3](=[OX1])[FX1,ClX1,BrX1,IX1]',
        'Carbonate': '[#6][OX2][CX3](=[OX1])[OX2][#6]',
        'Carbamate': '[NX3][CX3](=[OX1])[OX2][#6]',
        'Ether': '[OD2]([#6])[#6]',
        'Peroxide': '[OX2][OX2]',
        'Epoxide': '[OX2r3]1[#6r3][#6r3]1',
        'Hydroxamic Acid': '[CX3](=[OX1])[NX3][OX2H]',

        # Nitrogen-containing groups
        'Amine (primary)': '[NX3;H2][#6]',
        'Amine (secondary)': '[NX3;H1]([#6])[#6]',
        'Amine (tertiary)': '[NX3]([#6])([#6])[#6]',
        'Amide': '[NX3][CX3](=[OX1])',
        'Primary Amide': '[NX3H2][CX3](=[OX1])',
        'Secondary Amide': '[NX3H1]([#6])[CX3](=[OX1])',
        'Tertiary Amide': '[NX3]([#6])([#6])[CX3](=[OX1])',
        'Imine': '[NX2]=[CX3]',
        'Nitro': '[NX3](=[OX1])(=[OX1])',
        'Nitrile/Cyanide': '[NX1]#[CX2]',
        'Isocyanate': '[NX2]=[CX2]=[OX1]',
        'Isothiocyanate': '[NX2]=[CX2]=[SX1]',
        'Oxime': '[NX2]([OX2H])[CX3]',
        'Hydrazone': '[NX3H1][NX2]=[CX3]',
        'Azo': '[NX2]=[NX2]',
        'Nitrate': '[NX3](=[OX1])(=[OX1])[OX1]',
        'Nitrite': '[NX3](=[OX1])[OX1]',
        'Azide': '[NX1]=[NX2]=[NX2]',

        # Sulfur-containing groups
        'Thiol': '[SX2H]',
        'Sulfide/Thioether': '[#16X2]([#6])[#6]',
        'Disulfide': '[#16X2][#16X2]',
        'Sulfoxide': '[#16X3](=[OX1])([#6])[#6]',
        'Sulfone': '[#16X4](=[OX1])(=[OX1])([#6])[#6]',
        'Sulfonamide': '[#16X4](=[OX1])(=[OX1])([NX3])[#6]',
        'Sulfonic Acid': '[#16X4](=[OX1])(=[OX1])([OX2H])[#6]',
        'Sulfonate': '[#16X4](=[OX1])(=[OX1])([OX2][#6])[#6]',
        'Thioamide': '[#16X1][CX3]=[NX3]',

        # Phosphorus-containing groups
        'Phosphine': '[PX3]([#6])([#6])[#6]',
        'Phosphonic Acid': '[PX4](=[OX1])([OX2H])([OX2H])[#6]',
        'Phosphate': '[PX4](=[OX1])([OX2][#6])([OX2][#6])[OX2][#6]',
        'Phosphodiester': '[PX4](=[OX1])([OX2][#6])([OX2][#6])[OX2H,OX1-]',

        # Carbon structures
        'Alkene': '[CX3]=[CX3]',
        'Alkyne': '[CX2]#[CX2]',
        'Aromatic': 'a',
        'Benzene': 'c1ccccc1',
        'Phenyl': '[cX3]1[cX3][cX3][cX3][cX3][cX3]1',
        'Naphthalene': 'c1ccc2ccccc2c1',
        'Heterocycle': '[a;!c]',
        'Cycloalkane': '[CR3]1[CR2][CR2][CR2][CR2][CR2]1',
        'Cycloalkene': '[CR2]1[CR2][CR2][CR1]=[CR1][CR2]1',

        # Halogens
        'Halogen (F)': '[F]',
        'Halogen (Cl)': '[Cl]',
        'Halogen (Br)': '[Br]',
        'Halogen (I)': '[I]',

        # Boron-containing groups
        'Boronic Acid': '[BX3]([OX2H])[OX2H]',
        'Boronate Ester': '[BX3]([OX2][#6])[OX2][#6]',

        # Silicon-containing groups
        'Silane': '[SiX4]([#6])([#6])([#6])[#6]',
        'Silanol': '[SiX4]([#6])([#6])([#6])[OX2H]',
        'Silyl Ether': '[SiX4]([#6])([#6])([#6])[OX2][#6]',

        # Heterocycles
        'Pyridine': 'n1ccccc1',
        'Pyrimidine': 'n1cnccc1',
        'Pyrrole': '[nH]1cccc1',
        'Furan': 'o1cccc1',
        'Thiophene': 's1cccc1',
        'Imidazole': 'n1cncc1',
        'Pyrazole': 'n1nccn1',
        'Oxazole': 'o1cncc1',
        'Thiazole': 's1cncc1',
        'Piperidine': 'N1CCCCC1',
        'Piperazine': 'N1CCNCC1',
        'Morpholine': 'O1CCNCC1',
        'Indole': 'c1ccc2[nH]ccc2c1',
        'Quinoline': 'n1cccc2ccccc12',
    }

    # Compile the SMARTS patterns
    pattern_dict = {}
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            pattern_dict[name] = pattern

    # Process each molecule
    results = []

    for i, mol in enumerate(mol_list):
        if mol is None:
            print(f"Warning: Molecule at index {i} is None")
            continue

        # Create a dictionary to store functional groups and their atom indices
        fg_dict = {}

        for name, pattern in pattern_dict.items():
            # Get all matches
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                # Store list of atom indices for each match
                fg_dict[name] = [list(match) for match in matches]

        # Add the result tuple
        results.append((mol, fg_dict))

    return results

# Example usage
def example():
    # Create some example molecules
    smiles_list = [
        'CCO',  # ethanol
        'CC(=O)C',  # acetone
        'CC(=O)O',  # acetic acid
        'c1ccccc1',  # benzene
        'CC(=O)NC',  # N-methylacetamide
        'CCN',  # ethylamine
        'CC#N'   # acetonitrile
    ]

    # Convert SMILES to RDKit molecules
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # We need to ensure hydrogens are explicitly represented
            mol = Chem.AddHs(mol)
            mols.append(mol)

    # Extract functional groups
    results = extract_functional_groups(mols)

    # Print results
    for mol, fg_dict in results:
        print(f"Molecule: {Chem.MolToSmiles(mol)}")
        if fg_dict:
            for group, matches in fg_dict.items():
                print(f"  {group}: {matches}")
        else:
            print("  No functional groups found")
        print()

    return results

# For molecules without hydrogens, as specified in the requirement
def use_with_no_hydrogen_mols():
    # Create some example molecules
    smiles_list = [
        'CCO',  # ethanol
        'CC(=O)C',  # acetone
        'CC(=O)O',  # acetic acid
        'c1ccccc1',  # benzene
        'CC(=O)NC',  # N-methylacetamide
        'CCN',  # ethylamine
        'CC#N'   # acetonitrile
    ]

    # Convert SMILES to RDKit molecules without adding hydrogens
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        # Do NOT add hydrogens, as per requirement
        if mol:
            mols.append(mol)

    # Extract functional groups
    results = extract_functional_groups(mols)

    # Print results - adjusted for molecules without hydrogens
    for mol, fg_dict in results:
        print(f"Molecule: {Chem.MolToSmiles(mol)}")
        if fg_dict:
            for group, matches in fg_dict.items():
                print(f"  {group}: {matches}")
        else:
            print("  No functional groups found")
        print()

    return results

# Run the example if this script is executed directly
if __name__ == "__main__":
    print("Example with molecules WITH explicit hydrogens:")
    example()
    print("\nExample with molecules WITHOUT explicit hydrogens (as per requirement):")
    use_with_no_hydrogen_mols()