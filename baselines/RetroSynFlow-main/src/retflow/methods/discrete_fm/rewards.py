from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from rdkit import Chem
from rdkit.Chem import QED

from retflow.utils import get_forward_model, smi_tokenizer


@dataclass
class Reward(ABC):
    @abstractmethod
    def initialize_reward(self):
        pass

    @abstractmethod
    def compute_reward(self, x1_smiles, prod_smiles, device):
        pass


@dataclass
class QEDReward(Reward):
    def initialize_reward(self):
        pass

    def compute_reward(self, x1_mols, prod_mols, device):
        rewards = []
        for x1_mol in x1_mols:
            try:
                score = QED.qed(x1_mol)
            except Chem.rdchem.MolSanitizeException:
                score = -1.0
            rewards.append(score)

        return torch.tensor(rewards).to(device)


@dataclass
class ForwardSynthesisReward(Reward):
    n_best: int = 5

    def initialize_reward(self):
        self.forward_translator = get_forward_model(self.n_best)

    def compute_reward(self, x1_smiles, prod_smiles, device):
        tokenized_smiles = [smi_tokenizer(x1.strip()) for x1 in x1_smiles]

        _, pred_products = self.forward_translator.translate(
            src_data_iter=tokenized_smiles,
            batch_size=256,
            attn_debug=False,
        )
        rewards = []
        for i, predictions in enumerate(pred_products):
            preds = ["".join(p.split()) for p in predictions]
            reward = preds.count(prod_smiles[i]) / self.n_best
            rewards.append(reward)
        return torch.tensor(rewards).to(device)


@dataclass
class ForwardSynthesisBinaryReward(Reward):
    n_best: int = 5

    def initialize_reward(self):
        self.forward_translator = get_forward_model(self.n_best)

    def compute_reward(self, x1_smiles, prod_smiles, device):
        tokenized_smiles = [smi_tokenizer(x1.strip()) for x1 in x1_smiles]

        _, pred_products = self.forward_translator.translate(
            src_data_iter=tokenized_smiles,
            batch_size=256,
            attn_debug=False,
        )
        rewards = []
        for i, predictions in enumerate(pred_products):
            preds = ["".join(p.split()) for p in predictions]
            reward = 1 if prod_smiles[i] in preds else 0
            rewards.append(reward)
        return torch.tensor(rewards).to(device)
