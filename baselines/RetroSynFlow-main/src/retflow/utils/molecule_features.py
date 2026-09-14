import torch

from retflow.utils.wrappers import GraphWrapper

USPTO_VALENCIES = [
    5, 4, 6, 6, 7, 1, 3, 7, 5, 4, 7, 4, 2, 4, 2, 6,
    3, 8, 5, 4, 1, 1, 1, 10, 8, 3, 0,
]

USPTO_ATOM_WEIGHTS = {
    1: 14.01,
    2: 12.01,
    3: 16.0,
    4: 32.06,
    5: 35.45,
    6: 19.0,
    7: 10.81,
    8: 79.91,
    9: 30.98,
    10: 28.01,
    11: 126.9,
    12: 118.71,
    13: 24.31,
    14: 63.55,
    15: 65.38,
    16: 78.97,
    17: 26.98,
    18: 39.95,
    19: 74.92,
    20: 72.63,
    21: 39.10,
    22: 6.94,
    23: 22.99,
    24: 106.42,
    25: 101.07,
    26: 44.96,
    27: 0.0,
}


class ExtraMolecularFeatures:
    def __init__(self):
        self.charge = ChargeFeature(remove_h=True, valencies=USPTO_VALENCIES)
        self.valency = ValencyFeature()
        self.weight = WeightFeature(max_weight=1000, atom_weights=USPTO_ATOM_WEIGHTS)

    def __call__(self, X, E, node_mask):
        charge = self.charge(X, E, node_mask).unsqueeze(-1)  # (bs, n, 1)
        valency = self.valency(X, E, node_mask).unsqueeze(-1)  # (bs, n, 1)
        weight = self.weight(X, E, node_mask)  # (bs, 1)

        extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)

        return GraphWrapper(
            X=torch.cat((charge, valency), dim=-1), E=extra_edge_attr, y=weight
        )


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, X, E, node_mask):
        bond_orders = torch.tensor(
            [0, 1, 2, 3, 1.5], device=E.device
        ).reshape(1, 1, 1, -1)
        weighted_E = E * bond_orders  # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)  # (bs, n)

        valencies = torch.tensor(
            self.valencies, device=X.device
        ).reshape(1, 1, -1)
        X = X * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)  # (bs, n)

        return (normal_valencies - current_valencies).type_as(X)


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, X, E, node_mask):
        orders = torch.tensor(
            [0, 1, 2, 3, 1.5], device=E.device
        ).reshape(1, 1, 1, -1)
        E = E * orders  # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)  # (bs, n)
        return valencies.type_as(X)


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.tensor(list(atom_weights.values()))

    def __call__(self, X, E, node_mask):
        X = torch.argmax(X, dim=-1)  # (bs, n)
        atom_weight_list = self.atom_weight_list.to(X.device)
        X_weights = atom_weight_list[X]  # (bs, n)
        return (
            X_weights.sum(dim=-1).unsqueeze(-1).type_as(X)
            / self.max_weight
        )
