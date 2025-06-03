from rdkit import Chem
import numpy as np

class FGFeaturizer:
    def __init__(self):
        with open('/home/nobilm@usi.ch/pretrain_paper/source/scripts/fg.txt', "r") as f:
            self.funcgroups = f.read().strip().split('\n')
            self.name = [i.split()[0] for i in self.funcgroups]
            self.smarts = [Chem.MolFromSmarts(i.split()[1]) for i in self.funcgroups]
            self.name2int = {self.name[i]: i for i in range(len(self.name))}
            self.number_fg = len(self.name) + 2  # +1 for the "no functional group" case; +1 for masking

    def process_mol(self, m):
        fg_matches = []
        for fg_idx, fg_mol in enumerate(self.smarts):
            matches = m.GetSubstructMatches(fg_mol,
                                            useChirality=True,
                                            useQueryQueryMatches=True,
                                            # uniquify=False,
                                            )
            if matches:
                fg_name = self.name[fg_idx]
                fg_matches.append({'name': fg_name, 'atom_indices': matches})

        # Map atom index to all functional group names it belongs to
        atom2fgname = {}
        for fg in fg_matches:
            for match in fg['atom_indices']:
                for idx in match:
                    atom2fgname.setdefault(idx, set()).add(fg['name'])

        self.atom_fgs=atom2fgname

    def fg_per_atom(self, atom_idx):
        fg_ids = np.zeros(self.number_fg, dtype=int)
        ones_for_this_atom = []
        if atom_idx in self.atom_fgs and self.atom_fgs[atom_idx] is not None:
            fg_names = list(self.atom_fgs[atom_idx]) # get list of fg names associated to this atom
            for name in fg_names:
                fg_id = self.name2int[name]
                ones_for_this_atom.append(fg_id)
        else:
            ones_for_this_atom = [-2]
        fg_ids[ones_for_this_atom] = 1
        return fg_ids
