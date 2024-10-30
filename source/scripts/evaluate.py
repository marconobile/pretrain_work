import os
import sys
import argparse
import logging
from typing import Union
from pathlib import Path
import numpy as np
import h5py
from torch_scatter import scatter

from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset
from geqtrain.data import AtomicDataDict
from geqtrain.data._build import dataset_from_config
from geqtrain.data.dataloader import DataLoader
from geqtrain.scripts.deploy import load_deployed_model, CONFIG_KEY
from geqtrain.train import Trainer
from geqtrain.train.metrics import Metrics
from geqtrain.train.trainer import run_inference, remove_node_centers_for_NaN_targets
from geqtrain.utils import Config
from geqtrain.utils.auto_init import instantiate
from geqtrain.utils.savenload import load_file

from sklearn.metrics import confusion_matrix

def main(args=None, running_as_script: bool = True):
    # in results dir, do: geqtrain-deploy build --train-dir . deployed.pth
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "-td",
        "--train-dir",
        help="Path to a working directory from a training session.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="A deployed or pickled GEqTrain model to load. If omitted, defaults to `best_model.pth` in `train_dir`.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-tc",
        "--test-config",
        help="A YAML config file specifying the dataset to load test data from. If omitted, `config.yaml` in `train_dir` will be used",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        help="Batch size to use. Larger is usually faster on GPU. If you run out of memory, lower this. You can also try to raise this for faster evaluation. Default: 16.",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default='cpu',
    )
    parser.add_argument(
        "--test-indexes",
        help="Path to a file containing the indexes in the dataset that make up the test set. "
             "If omitted, all data frames *not* used as training or validation data in the training session `train_dir` will be used.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--stride",
        help="If dataset config is provided and test indexes are not provided, take all dataset idcs with this stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        help="Try to have PyTorch use deterministic algorithms. Will probably fail on GPU/CUDA.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--log",
        help="log file to store all the metrics and screen logging.debug",
        type=Path,
        default=None,
    )
    # parser.add_argument(
    #     "--output",
    #     help="ExtXYZ (.xyz) file to write out the test set and model predictions to.",
    #     type=Path,
    #     default=None,
    # )
    # parser.add_argument(
    #     "--output-fields",
    #     help="Extra fields (names[:field] comma separated with no spaces) to write to the `--output`.\n"
    #          "Field options are: [node, edge, graph, long].\n"
    #          "If [:field] is omitted, the field with that name is assumed to be already registered by default.",
    #     type=str,
    #     default="",
    # )
        # parser.add_argument(
    #     "--repeat",
    #     help=(
    #         "Number of times to repeat evaluating the test dataset. "
    #         "This can help compensate for CUDA nondeterminism, or can be used to evaluate error on models whose inference passes are intentionally nondeterministic. "
    #         "Note that `--repeat`ed passes over the dataset will also be `--output`ed if an `--output` is specified."
    #     ),
    #     type=int,
    #     default=1,
    # )

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    # Parse args
    args = parser.parse_args(args=args)

    # Do the defaults:
    dataset_is_from_training: bool = False
    print_best_model_epoch: bool = False
    if args.train_dir:
        if args.test_config is None:
            args.test_config = args.train_dir / "config.yaml"
            dataset_is_from_training = True
        if args.model is None:
            print_best_model_epoch = True
            args.model = args.train_dir / "best_model.pth"
        if args.test_indexes is None and dataset_is_from_training:
            # Find the remaining indexes that aren't train or val
            trainer = torch.load(
                str(args.train_dir / "trainer.pth"), map_location="cpu"
            )
            if print_best_model_epoch:
                print(f"Loading model from epoch: {trainer.best_model_saved_at_epoch}")
            train_idcs = []
            dataset_offset = 0
            for tr_idcs in trainer["train_idcs"]:
                train_idcs.extend([tr_idx + dataset_offset for tr_idx in tr_idcs.tolist()])
                dataset_offset += len(tr_idcs)
            train_idcs = set(train_idcs)
            val_idcs = []
            dataset_offset = 0
            for v_idcs in trainer["val_idcs"]:
                val_idcs.extend([v_idx + dataset_offset for v_idx in v_idcs.tolist()])
                dataset_offset += len(v_idcs)
            val_idcs = set(val_idcs)
        else:
            train_idcs = val_idcs = None

    # validate
    if args.test_config is None:
        raise ValueError("--test-config or --train-dir must be provided")

    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device) # must be cuda:dev_id

    # logger
    logger = logging.getLogger("geqtrain-evaluate")
    logger.setLevel(logging.INFO)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(
            "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
        )

    if args.use_deterministic_algorithms:
        logger.info(
            "Telling PyTorch to try to use deterministic algorithms... please note that this will likely error on CUDA/GPU"
        )
        torch.use_deterministic_algorithms(True)

    # Load model
    model, config = load_model(args.model, device=args.device)

    # Load config file
    logger.info(
        f"Loading {'training' if dataset_is_from_training else 'test'} dataset...",
    )

    evaluate_config = Config.from_file(str(args.test_config), defaults={})
    config.update(evaluate_config)

    dataset_is_test: bool = False
    dataset_is_validation: bool = False
    try:
        # Try to get test dataset
        dataset = dataset_from_config(config, prefix="test_dataset")
        dataset_is_test = True
    except KeyError:
        pass
    if not dataset_is_test:
        try:
            # Try to get validation dataset
            dataset = dataset_from_config(config, prefix="validation_dataset")
            dataset_is_validation = True
        except KeyError:
            pass

    if not (dataset_is_test or dataset_is_validation):
        raise Exception("Either test or validation dataset must be provided.")
    logger.info(
        f"Loaded {'test_' if dataset_is_test else 'validation_' if dataset_is_validation else ''}dataset specified in {args.test_config.name}.",
    )

    if args.test_indexes is None:
        # Default to all frames
        test_idcs = [torch.arange(len(ds)) for ds in dataset.datasets]
        logger.info(
            f"Using all frames from the specified test dataset with stride {args.stride}, yielding a test set size of {len(test_idcs)} frames.",
        )
    else:
        # load from file
        test_idcs = load_file(
            supported_formats=dict(
                torch=["pt", "pth"], yaml=["yaml", "yml"], json=["json"]
            ),
            filename=str(args.test_indexes),
        )
        logger.info(
            f"Using provided test set indexes, yielding a test set size of {len(test_idcs)} frames.",
        )

    test_idcs = [torch.as_tensor(idcs, dtype=torch.long)[::args.stride] for idcs in test_idcs]
    # test_idcs = test_idcs.tile((args.repeat,))

    # Figure out what metrics we're actually computing
    try:
        metrics_components = config.get("metrics_components", None)
        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=config,
        )
        metrics.to(device=device)
        metrics_metadata = {
            'type_names'   : config["type_names"],
            'target_names' : config.get('target_names', list(metrics.keys)),
        }
    except:
        raise Exception("Failed to load Metrics.")

    # --- filter node target to train on based on node type or type name
    keep_type_names = config.get("keep_type_names", None)
    if keep_type_names is not None:
        from geqtrain.train.utils import find_matching_indices
        keep_node_types = torch.tensor(find_matching_indices(config["type_names"], keep_type_names))
    else:
        keep_node_types = None

    # dataloader
    per_node_outputs_keys = None
    _indexed_datasets = []
    for _dataset, _test_idcs in zip(dataset.datasets, test_idcs):
        _dataset = _dataset.index_select(_test_idcs)
        _dataset, per_node_outputs_keys = remove_node_centers_for_NaN_targets(_dataset, metrics, keep_node_types)
        if _dataset is not None:
            _indexed_datasets.append(_dataset)
    dataset_test = ConcatDataset(_indexed_datasets)

    dataloader = DataLoader(
        dataset=dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
    )

    # run inference
    logger.info("Starting...")
    conf_matrix = np.array([0.])


    def metrics_callback(pbar, out, ref_data):
        # accumulate metrics
        batch_metrics = metrics(pred=out, ref=ref_data)

        desc = '\t'.join(
            f'{k:>20s} = {v:< 20f}'
            for k, v in metrics.flatten_metrics(
                batch_metrics,
                metrics_metadata=metrics_metadata,
            )[0].items()
        )
        pbar.set_description(f"Metrics: {desc}")
        del out, ref_data

    config.pop('device')
    save_out_feature = True # TODO: extract this in args
    observations, gt = infer(dataloader, model, device, per_node_outputs_keys, chunk_callbacks=[metrics_callback], save_out_feature=save_out_feature, **config) # chunk == batch

    if save_out_feature:
        filename = './train_data.h5' # TODO: extract this in args + add handling of the .h5 extension
        write_batched_obs_to_file(len(dataloader), filename, observations, gt)

    logger.info("\n--- Final result: ---")
    logger.info(
        "\n".join(
            f"{k:>20s} = {v:< 20f}"
            for k, v in metrics.flatten_metrics(
                metrics.current_result(),
                metrics_metadata=metrics_metadata,
            )[0].items()
        )
    )


# def write_obs_to_file(filename:str='./dataset.h5', observations:List=[], ground_truths:List=[]):
#     with h5py.File(filename, 'w') as h5_file:
#         for i, (obs, gt) in enumerate(zip(observations, ground_truths)):
#             h5_file.create_dataset(f'observation_{i}', data=obs.numpy())
#             h5_file.create_dataset(f'ground_truth_{i}', data=gt.numpy())

#     # reopen to test consistency
#     hdf5_file = h5py.File(filename, 'r')
#     assert len(hdf5_file.keys())//2 == len(observations)

def write_batched_obs_to_file(n_batches, filename:str='./dataset.h5', observations:List=[], ground_truths:List=[]):
    dset_id = 0
    with h5py.File(filename, 'w') as h5_file:
        for batch_idx in range(n_batches):
            obs_batch, gt_batch = observations[batch_idx], ground_truths[batch_idx]
            for obs_idx in range(obs_batch.shape[0]): # expected bs first
                obs, gt = obs_batch[obs_idx], gt_batch[obs_idx]
                h5_file.create_dataset(f'observation_{dset_id}', data=obs.numpy())
                h5_file.create_dataset(f'ground_truth_{dset_id}', data=gt.numpy())
                dset_id+=1


def infer(dataloader, model, device, per_node_outputs_keys, chunk_callbacks=[], batch_callbacks=[], save_out_feature:bool=False, **kwargs):
    pbar = tqdm(dataloader)
    observations, gt = [] , []
    for batch_index, data in enumerate(pbar):
        already_computed_nodes = None
        while True:
            out, ref_data, batch_chunk_center_nodes, num_batch_center_nodes = run_inference(
                model=model,
                data=data,
                device=device,
                cm=torch.no_grad(),
                already_computed_nodes=already_computed_nodes,
                per_node_outputs_keys=per_node_outputs_keys,
                **kwargs,
            )

            if save_out_feature:
                field = AtomicDataDict.NODE_FEATURES_KEY
                out_field = AtomicDataDict.GRAPH_OUTPUT_KEY
                graph_feature = scatter(out[field][...,:32], index = out['batch'], dim=0)
                # _, counts = out['batch'].unique(return_counts=True)
                # graph_feature /=counts
                observations.append(graph_feature.cpu())
                gt.append(out[out_field].cpu())

            for callback in chunk_callbacks:
                callback(pbar, out, ref_data, **kwargs)

            target = ref_data["graph_output"].cpu().bool()
            prediction = (out["graph_output"].sigmoid()>.5).cpu().bool()

            if np.sum(conf_matrix) == 0:
                conf_matrix = confusion_matrix(target, prediction)
            else:
                conf_matrix += confusion_matrix(target, prediction)

            # evaluate ending condition
            if already_computed_nodes is None: # already_computed_nodes is the stopping criteria to finish batch step
                if len(batch_chunk_center_nodes) < num_batch_center_nodes:
                    already_computed_nodes = batch_chunk_center_nodes
            elif len(already_computed_nodes) + len(batch_chunk_center_nodes) == num_batch_center_nodes:
                already_computed_nodes = None
            else:
                assert len(already_computed_nodes) + len(batch_chunk_center_nodes) < num_batch_center_nodes
                already_computed_nodes = torch.cat([already_computed_nodes, batch_chunk_center_nodes], dim=0)

            if already_computed_nodes is None:
                break

        for callback in batch_callbacks:
            callback(batch_index, **kwargs)

    print(conf_matrix)
    tn, fp, fn, tp = conf_matrix.ravel()
    print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: ", tp)

    return observations, gt


def load_model(model: Union[str, Path], device="cpu"):
    if isinstance(model, str):
        model = Path(model)
    logger = logging.getLogger("geqtrain-evaluate")
    logger.setLevel(logging.INFO)

    logger.info("Loading model... ")

    try:
        model, metadata = load_deployed_model(
            model,
            device=device,
            set_global_options=True,  # don't warn that setting
        )
        logger.info("loaded deployed model.")

        import tempfile
        tmp = tempfile.NamedTemporaryFile()
        # Open the file for writing.
        with open(tmp.name, 'w') as f:
            f.write(metadata[CONFIG_KEY])
        model_config = Config.from_file(tmp.name)

        model.eval()
        return model, model_config
    except ValueError:  # its not a deployed model
        pass

    # load a training session model
    model, model_config = Trainer.load_model_from_training_session(
        traindir=model.parent, model_name=model.name, device=device
    )
    logger.info("loaded model from training session.")
    model.eval()

    return model, model_config

if __name__ == "__main__":
    main(running_as_script=True)