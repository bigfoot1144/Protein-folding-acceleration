import re
import torch
from data import Alphabet
import torch
from esm2 import ESM2
import random

SEQ_FILE = "protein_sequences.txt"  # File you just created
TOP_K = 5                          # Top-K predictions to consider for accuracy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "esm2_t36_3B_UR50D"

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

# Load the main model data
model_data = torch.load(f'../model/{MODEL_NAME}.pt', 
                        mmap=True, weights_only=False)
print("Main model loaded")

# Get model, alphabet, and upgraded state_dict
model, alphabet, state_dict = _load_model_and_alphabet_core_v2(model_data)

# Load main model weights (without contact head)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"Loaded main model. Missing keys: {missing_keys}")

# Load contact head weights from separate file
contact_data = torch.load(f'../model/{MODEL_NAME}-contact-regression.pt', 
                          mmap=True, weights_only=False)
print("Contact head data loaded")

# Extract contact head state dict
contact_state_dict = contact_data["model"]

# Remove any prefixes (adjust based on what keys look like)
# Check the actual keys first to see what prefix needs removing
print("Contact head keys:", list(contact_state_dict.keys())[:5])

# Common patterns to try:
# Option A: If keys have "contact_head." prefix
contact_state_dict_cleaned = {}
for k, v in contact_state_dict.items():
    # Remove common prefixes
    new_key = k.replace("encoder.sentence_encoder.", "")
    new_key = new_key.replace("encoder.", "")
    new_key = new_key.replace("contact_head.", "")
    contact_state_dict_cleaned[new_key] = v

# Load into contact head
model.contact_head.load_state_dict(contact_state_dict_cleaned, strict=True)
print("Contact head loaded successfully")

# Set to evaluation mode
model.to(DEVICE).eval()

print("Model ready for inference!")

# Load sequences from file
sequences = []
seq_names = []

with open(SEQ_FILE, "r") as f:
    for i, line in enumerate(f):
        seq = line.strip().upper()
        if len(seq) > 5:  # skip very short sequences
            sequences.append(seq)
            seq_names.append(f"seq_{i}")

print(f"Loaded {len(sequences)} sequences from {SEQ_FILE}")

# Mask a random position per sequence
masked_sequences = []
true_residues = []

for seq in sequences:
    # Only choose positions that are standard tokens
    valid_positions = [i for i, aa in enumerate(seq) if aa in alphabet.standard_toks]
    if not valid_positions:
        continue
    mask_pos = random.choice(valid_positions)
    true_residues.append(seq[mask_pos])
    masked_seq = seq[:mask_pos] + alphabet.get_tok(alphabet.mask_idx) + seq[mask_pos+1:]
    masked_sequences.append(masked_seq)

# Prepare batch tensor (on GPU)
batch = list(zip(seq_names, masked_sequences))
_, _, tokens = alphabet.get_batch_converter()(batch)
tokens = tokens.to(DEVICE)
model.to(DEVICE)
print(f"Tokenized batch shape: {tokens.shape} on {DEVICE}")

# Forward pass
with torch.no_grad():
    out = model(tokens, repr_layers=[33], return_contacts=False)

logits = out["logits"]  # (batch, seq_len, vocab_size)

# valuate masked predictions
correct_top1 = 0
correct_topk = 0
total = len(masked_sequences)

for b_idx in range(total):
    # Locate mask position
    mask_positions = (tokens[b_idx] == alphabet.mask_idx).nonzero(as_tuple=False)
    if len(mask_positions) == 0:
        continue
    pos_i = mask_positions[0, 0].item()
    print(pos_i)

    # Get logits for the masked token
    mask_logits = logits[b_idx, pos_i]
    probs = torch.softmax(mask_logits, dim=0)

    topk_probs, topk_indices = torch.topk(probs, TOP_K)
    topk_tokens = [alphabet.get_tok(idx.item()) for idx in topk_indices]

    true_res = true_residues[b_idx]

    # Accuracy
    if topk_tokens[0] == true_res:
        correct_top1 += 1
    if true_res in topk_tokens:
        correct_topk += 1

print("\n========== Accuracy Summary ==========")
print(f"Top-1 accuracy: {correct_top1}/{total} = {correct_top1/total:.2f}")
print(f"Top-{TOP_K} accuracy: {correct_topk}/{total} = {correct_topk/total:.2f}")