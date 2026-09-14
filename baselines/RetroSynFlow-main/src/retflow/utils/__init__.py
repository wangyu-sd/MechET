from retflow.utils.data import (build_molecule, build_simple_molecule,
                                get_graph_list, get_molecule_list,
                                get_molecule_smi_list, get_synthons,
                                reactants_with_partial_atom_mapping, to_dense)
from retflow.utils.graph_features import ExtraFeatures
from retflow.utils.molecule_features import ExtraMolecularFeatures
from retflow.utils.wrappers import (GraphDimensions, GraphModelLayerInfo,
                                    GraphModelWrapper, GraphWrapper)


def top_k_accuracy(*args, **kwargs):
    from retflow.utils.eval_helper import top_k_accuracy as implementation

    return implementation(*args, **kwargs)


def get_forward_model(*args, **kwargs):
    from retflow.utils.forward_model import get_forward_model as implementation

    return implementation(*args, **kwargs)


def smi_tokenizer(*args, **kwargs):
    from retflow.utils.forward_model import smi_tokenizer as implementation

    return implementation(*args, **kwargs)
