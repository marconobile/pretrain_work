# code modified from https://www.kaggle.com/code/hengck23/conforge-open-source-conformer-generator
import sys
import CDPL.Chem as CDPLChem
import CDPL.ConfGen as ConfGen
from tqdm import tqdm

# !pip install cdpkit


# dictionary mapping status codes to human readable strings
status_to_str = {
    ConfGen.ReturnCode.UNINITIALIZED: 'uninitialized',
    ConfGen.ReturnCode.TIMEOUT: 'max. processing time exceeded',
    ConfGen.ReturnCode.ABORTED: 'aborted',
    ConfGen.ReturnCode.FORCEFIELD_SETUP_FAILED: 'force field setup failed',
    ConfGen.ReturnCode.FORCEFIELD_MINIMIZATION_FAILED: 'force field structure refinement failed',
    ConfGen.ReturnCode.FRAGMENT_LIBRARY_NOT_SET: 'fragment library not available',
    ConfGen.ReturnCode.FRAGMENT_CONF_GEN_FAILED: 'fragment conformer generation failed',
    ConfGen.ReturnCode.FRAGMENT_CONF_GEN_TIMEOUT: 'fragment conformer generation timeout',
    ConfGen.ReturnCode.FRAGMENT_ALREADY_PROCESSED: 'fragment already processed',
    ConfGen.ReturnCode.TORSION_DRIVING_FAILED: 'torsion driving failed',
    ConfGen.ReturnCode.CONF_GEN_FAILED: 'conformer generation failed'
}

def gen3DStructure(mol: CDPLChem.Molecule, struct_gen: ConfGen.StructureGenerator) -> int:
    # prepare the molecule for 3D structure generation
    ConfGen.prepareForConformerGeneration(mol)

    # generate the 3D structure
    status = struct_gen.generate(mol)

    # if successful, store the generated conformer ensemble as
    # per atom 3D coordinates arrays (= the way conformers are represented in CDPKit)

    if status == ConfGen.ReturnCode.SUCCESS:
        struct_gen.setCoordinates(mol)

    # return status code
    return status

def generate_conf_sdf(inpt_smiles, sdf_filepath):
  '''
  sdf_filepath must be a path/to/file.sdf str
  '''
  print(f'Start writing {sdf_filepath}')
  # export only a single 3D structure (in case of multi-conf. input molecules)
  writer = CDPLChem.MolecularGraphWriter(sdf_filepath)
  CDPLChem.setMultiConfExportParameter(writer, False)

  # create and initialize an instance of the class ConfGen.StructureGenerator which will
  # perform the actual 3D structure generation work
  struct_gen = ConfGen.StructureGenerator()
  struct_gen.settings.timeout = 3600 * 1000  # apply the -t argument

  # create an instance of the default implementation of the CDPLChem.Molecule interface
  mol = CDPLChem.BasicMolecule()

  for t,s in enumerate(tqdm(inpt_smiles)):

      # compose a simple molecule identifier
      mol_id = f'Mol_{t}'
      mol = CDPLChem.parseSMILES(s)

      #print('- Generating 3D structure of molecule %s...' % mol_id)
      try:

          # generate 3D structure of the read molecule
          status = gen3DStructure(mol, struct_gen)

          # check for severe error reported by status code
          if status != ConfGen.ReturnCode.SUCCESS:
                  print('Error: 3D structure generation for molecule %s failed: %s' % (
                  mol_id, status_to_str[status]))

          else:

              # enforce the output of 3D coordinates in case of MDL file formats
              CDPLChem.setMDLDimensionality(mol, 3)

              # output the generated 3D structure
              if not writer.write(mol):
                  sys.exit('Error: writing 3D structure of molecule %s failed' % mol_id)


      except Exception as e:
          sys.exit('Error: 3D structure generation or output for molecule %s failed: %s' % (mol_id, str(e)))

  writer.close()
  print('Write ended ok')