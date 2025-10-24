import typing as T
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from torch import nn
from torch.nn import LayerNorm

from data import Alphabet
from categorical_mixture import categorical_lddt
from trunk import FoldingTrunk, FoldingTrunkConfig
import residue_constants

from typing import Dict, Optional, Tuple

def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers

def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )

def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    bin_centers = _calculate_bin_centers(boundaries)
    torch.sum(residue_weights)
    n = logits.shape[-2]
    clipped_n = max(n, 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights
     
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]

def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_aligned_error,
        max_predicted_aligned_error,
    ) = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }

import residue_constants as rc

def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein['aatype'].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=protein["aatype"].device
    )
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


from esm2 import ESM2
import re

def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from torch import nn
from torch.nn import LayerNorm

from data import Alphabet
from categorical_mixture import categorical_lddt
from trunk import FoldingTrunk, FoldingTrunkConfig


@dataclass
class ESMFoldConfig:
    trunk: T.Any = FoldingTrunkConfig
    lddt_head_hid_dim: int = 128

class ESMFold(nn.Module):
    def __init__(self, esm, esm_dict,  esmfold_config=None, **kwargs):
        super().__init__()

        self.cfg = esmfold_config if esmfold_config else ESMFoldConfig(**kwargs)
        cfg = self.cfg

        self.distogram_bins = 64

        self.esm = esm
        self.esm_dict = esm_dict 

        self.esm.requires_grad_(False)
        self.esm.half()

        self.esm_feats = self.esm.embed_dim
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMFold._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim

        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )
        if cfg.use_esm_attn_map:
            self.esm_z_mlp = nn.Sequential(
                LayerNorm(self.esm_attns),
                nn.Linear(self.esm_attns, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
            )

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = FoldingTrunk(**cfg.trunk)

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(cfg.trunk.structure_module.c_s),
            nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in residue_constants.restypes_with_x
        ]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def _compute_language_model_representations(
        self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=self.cfg.use_esm_attn_map,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        esm_z = (
            res["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
            if self.cfg.use_esm_attn_map
            else None
        )
        return esm_s, esm_z

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

    def forward(
        self,
        aa: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """

        if mask is None:
            mask = torch.ones_like(aa)

        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        esmaa = self._af2_idx_to_esm_idx(aa, mask)

        if masking_pattern is not None:
            esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern)

        esm_s, esm_z = self._compute_language_model_representations(esmaa)

        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)
        esm_s = esm_s.detach()

        # === preprocessing ===
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)

        s_s_0 = self.esm_s_mlp(esm_s)
        if self.cfg.use_esm_attn_map:
            esm_z = esm_z.to(self.esm_s_combine.dtype)
            esm_z = esm_z.detach()
            s_z_0 = self.esm_z_mlp(esm_z)
        else:
            s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)

        s_s_0 += self.embedding(aa)

        structure: dict = self.trunk(
            s_s_0, s_z_0, aa, residx, mask, no_recycles=num_recycles
        )
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
            ]
        }

        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aa
        make_atom14_masks(structure)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx

        lddt_head = self.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0], B, L, -1, self.lddt_bins
        )
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        structure["plddt"] = (
            100 * plddt
        )  # we predict plDDT between 0 and 1, scale to be between 0 and 100.

        ptm_logits = self.ptm_head(structure["s_z"])

        seqlen = mask.type(torch.int64).sum(1)
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = torch.stack(
            [
                compute_tm(
                    batch_ptm_logits[None, :sl, :sl],
                    max_bins=31,
                    no_bins=self.distogram_bins,
                )
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ]
        )
        structure.update(
            compute_predicted_aligned_error(
                ptm_logits, max_bin=31, no_bins=self.distogram_bins
            )
        )

        return structure
    
    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.esm_s_combine.device


MODEL_NAME = "esm2_t36_3B_UR50D"

# Load the main model data
model_data = torch.load(f'../model/{MODEL_NAME}.pt', 
                        mmap=True, weights_only=False)
print("Main model loaded")

# Get model, alphabet, and upgraded state_dict
esm, alphabet, esm_dict = _load_model_and_alphabet_core_v2(model_data)

ESM_FOLD_MODEL_NAME = "esm_fold_v1"

esm_fold_data = torch.load(f'../model/{ESM_FOLD_MODEL_NAME}.pt', 
                        mmap=True, weights_only=False)

cfg = esm_fold_data["cfg"]["model"]
model_state = esm_fold_data["model"]
model = ESMFold(esm, alphabet, esmfold_config=cfg)

expected_keys = set(model.state_dict().keys())
found_keys = set(model_state.keys())

missing_essential_keys = []
for missing_key in expected_keys - found_keys:
    if not missing_key.startswith("esm."):
        missing_essential_keys.append(missing_key)

if missing_essential_keys:
    print("MISSING ESSENTIAL KEYS")
    raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

model.load_state_dict(model_state, strict=False)

batch_size = 2
sequence_length = 100

model = model.to("cuda")

## Main amino acid sequence tensor - indices for amino acids
## Based on the docstring, these should match openfold.np.residue_constants.restype_order_with_x
## Typically amino acid indices range from 0-20 (20 standard amino acids + unknown/mask)
aa = torch.randint(0, 21, (batch_size, sequence_length)).to("cuda")
#
## Optional: create a mask (1 for valid positions, 0 for padding)
mask = torch.ones(batch_size, sequence_length).to("cuda")
#
## Optional: residue indices (if not provided, will be assumed contiguous)
residx = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1).to("cuda")
#
## Optional: masking pattern for ESMFold
masking_pattern = torch.zeros(batch_size, sequence_length).to("cuda")  # No masking
## Or to randomly mask some positions:
## masking_pattern = torch.bernoulli(torch.full((batch_size, sequence_length), 0.1))  # 10% masked
#
## Number of recycles (defaults to 3 if None)
num_recycles = 3
#
## Call the forward function
output = model.forward(
    aa=aa,
    mask=mask,
    residx=residx,
    masking_pattern=masking_pattern,
    num_recycles=num_recycles
)