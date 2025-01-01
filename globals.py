from source.utils.yaml_parser import yaml_to_dict
import CDPL.ConfGen as ConfGen

yaml_params_path = "/home/nobilm@usi.ch/pretrain_paper/write_params.yaml"
params = yaml_to_dict(yaml_params_path)

max_confs = params['max_confs']
n_confs_to_keep = params['n_confs_to_keep'] # num of confs to keep after generating max_confs
min_rmsd = params['min_rmsd']
e_window = params['e_window']
num_threads = params['num_threads'] # mp.cpu_count()

# mapping status codes to human readable strings
status_to_str = { ConfGen.ReturnCode.UNINITIALIZED                  : 'uninitialized',
                  ConfGen.ReturnCode.TIMEOUT                        : 'max. processing time exceeded',
                  ConfGen.ReturnCode.ABORTED                        : 'aborted',
                  ConfGen.ReturnCode.FORCEFIELD_SETUP_FAILED        : 'force field setup failed',
                  ConfGen.ReturnCode.FORCEFIELD_MINIMIZATION_FAILED : 'force field structure refinement failed',
                  ConfGen.ReturnCode.FRAGMENT_LIBRARY_NOT_SET       : 'fragment library not available',
                  ConfGen.ReturnCode.FRAGMENT_CONF_GEN_FAILED       : 'fragment conformer generation failed',
                  ConfGen.ReturnCode.FRAGMENT_CONF_GEN_TIMEOUT      : 'fragment conformer generation timeout',
                  ConfGen.ReturnCode.FRAGMENT_ALREADY_PROCESSED     : 'fragment already processed',
                  ConfGen.ReturnCode.TORSION_DRIVING_FAILED         : 'torsion driving failed',
                  ConfGen.ReturnCode.CONF_GEN_FAILED                : 'conformer generation failed' }
