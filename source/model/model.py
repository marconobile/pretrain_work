import logging
from typing import Optional

from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset
from geqtrain.model import update_config
from geqtrain.nn.EnsembleLayer import EnsembleAggregator, WeightedEnsembleAggregator
from geqtrain.utils import Config

from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    NodewiseReduce,
    InteractionModule,
    EmbeddingAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    EmbeddingGraphAttrs,
    ReadoutModule,
)


def appendNGNNLayers(config):

    N:int = config.get('gnn_layers', 2)
    modules = {}
    logging.info(f"--- Number of GNN layers {N}")

    # # attention on embeddings
    modules.update({
        "update_emb": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_ATTRS_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
            # resnet=True,
            num_heads=8, # this number must be a 0 reminder of the sum of catted nn.embedded features (node and edges)
        ))
    })

    for layer_idx in range(N-1):
        modules.update({
            f"update_{layer_idx}": (ReadoutModule, dict(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.NODE_FEATURES_KEY, # scalars only
                out_irreps=None, # outs tensor of same o3.irreps of out_field
                resnet=True,
            )),
        })

    modules.update({
        "global_node_pooling": (NodewiseReduce, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.GRAPH_FEATURES_KEY,
            # residual_field=AtomicDataDict.NODE_ATTRS_KEY,
        )),
    })
    return modules

def transformer(config:Config):
    logging.info("--- Building Global Graph Model")

    update_config(config)

    if 'node_attributes' in config:
        node_embedder = (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.NODE_ATTRS_KEY,
            attributes=config.get('node_attributes'),
        ))
    else:
        raise ValueError('Missing node_attributes in yaml')

    if 'edge_attributes' in config:
        edge_embedder = (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            attributes=config.get('edge_attributes'),
        ))
    else:
        edge_embedder = None
        logging.info("--- Working without edge_attributes")

    if 'graph_attributes' in config:
        graph_embedder = EmbeddingGraphAttrs
    else:
        graph_embedder = None
        logging.info("--- Working without graph_attributes")

    layers = {
        # -- Encode -- #
        "node_attrs":         node_embedder,
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
    }

    if edge_embedder != None:
        layers.update({"edge_attrs": edge_embedder})

    if graph_embedder != None:
        layers.update({"graph_attrs": graph_embedder})

    layers.update(appendNGNNLayers(config))

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )