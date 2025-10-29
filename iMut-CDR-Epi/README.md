# iMut-CDR-Epi

**iMut-CDR-Epi** is a target-aware extension of iMut-CDR for **epitope-conditioned CDR design**.

## What it is
- A masked-residue recovery model for antibody CDRs that **conditions on the antigen pocket**.
- Built on the iMut-CDR backbone (ESM-2 encoder + MLM/contrast/distill losses), with an added **Graph Transformer pocket encoder** and **token-level FiLM** to modulate ESM-2 token states before the LM head.

## How it works (high level)

1. **Antigen pocket → graph**
   - **Nodes:** one node per antigen residue (Cα).
   - **Edges:**
     - **Sequence:** connect consecutive residues within the same chain.
     - **KNN:** for each residue, connect to its **12** nearest residues by Cα–Cα distance (cross-chain allowed).
     - All edges are **bidirectional**; no self-loops.
   - **Edge features (per edge):**
     - `d_scaled = min(||Cα_i − Cα_j||, 20 Å) / 20` (float)
     - `type_id ∈ {1 = sequence, 0 = knn}` (int; embedded by the GT)
2. **Graph Transformer encoding**  
   Encode the pocket graph and take a CLS-style readout to form a **conditioning vector**.
3. **FiLM modulation of ESM-2**  
   Map the conditioning vector to **γ/β** and apply per-token **FiLM** to ESM-2 hidden states before the LM head.
4. **Training objective**  
   Same objectives as iMut-CDR (masked-LM over 20 AAs, sequence-level contrast, token self-distillation), now **conditioned on the pocket**.

## Inputs and outputs
- **Inputs**
  - Antibody CDR sequence (mask tokens at candidate sites).
  - Antigen pocket structure (graph derived from pocket PDB/NPZ).
- **Outputs**
  - Per-position amino-acid distribution over 20 canonical AAs reflecting **epitope compatibility**.
  - In iterative sampling, single-site proposals are applied step-by-step to produce a **mutated CDR sequence**.

## Why it helps
- **Target awareness**: mutations fit the local antigen surface.
- **Coord-free, stable**: distance-aware pocket encoder without coordinate updates.
- **Drop-in**: retains iMut-CDR’s tokenizer, masking rules, and sampler; adds the pocket path plus FiLM.

---

# Repository Scope

This repository provides the **inference/mutation application** for iMut-CDR-Epi:

- Build antigen **pocket graphs** from PDB (or reuse cached `.npz`).
- Run **FiLM-conditioned** masked recovery to propose mutations.
- **Iterative loop** (one-site-at-a-time) with rich logging and summaries.
- Flexible **position selection**: all non-`X`, first-N, or user-defined indices.

For training and dataset preparation, see: [link pending].

---

# Pretrained Checkpoints

- **Conditional model (FiLM)**: `runs_cond_film_token_item_split_v1/best.pt` — checkpoint trained with antigen pocket conditioning.  
  Download: [link pending]
- **ESM-2 (650M) weights**: local folder with model files from Hugging Face.  
  Model card: [link pending] (`facebook/esm2_t33_650M_UR50D`)

> Place `best.pt` somewhere accessible (e.g., repo root), and set `LOCAL_MODEL_DIR` in the script to your local ESM-2 folder.

---

# Environment Setup

Use **Conda** (`environment.yml`) or **pip** (`requirements.txt`). Choose one.

## Option A — Conda
```bash
conda env create -f environment.yml
conda activate imut-cdr
```
- GPU users: ensure your CUDA driver matches the `pytorch-cuda` version in `environment.yml`.
- CPU-only users: remove or adjust CUDA-specific lines per PyTorch docs.

## Option B — pip
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
- GPU users: install a torch build compatible with your CUDA toolchain.
- CPU-only example:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

# Quick Start

1. **Prepare model files**
   - Conditional checkpoint at `runs_cond_film_token_item_split_v1/best.pt` (or another path).
   - Download ESM-2-650M locally and set `LOCAL_MODEL_DIR` in the script.

2. **Place antigen pocket PDBs (examples)**
   ```text
   raw_data/
     6cbpP_antigen_poc.pdb
     8x0tA_antigen_poc.pdb
     8wsqF_antigen_poc.pdb
   ```

3. **Run the application**
   - Main script: `mutate_epi.py`.
   - It has two clearly marked edit zones at the **bottom of the file**:
     - **INTERFACE ZONE (edit me)**: PDB paths, CDR sequences, checkpoint paths, **position strategy**.
     - **PARAMETERS (edit me)**: sampling temperature, Top-K/Top-P, `ALPHA/BETA/GAMMA`, etc.

   ```bash
   python mutate_epi.py
   ```

4. **Inspect outputs**
   - Per-iteration diagnostics and final **mutation order** per sequence/variant.
   - Programmatic output shape:
     ```text
     Dict[key -> List[(mut_seq: str, history: List[dict])]]
     ```

---

# Data Inputs

You provide:
- **Antigen pockets**: PDBs centered on the binding pocket (suffix `_antigen_poc.pdb` recommended).  
  The script caches atom tables to `data/antigens/<key>.npz`.
- **CDR sequences**: Strings using uppercase one-letter AAs; `X` marks non-mutating separators/padding.

**Example definition (inside `mutate_epi.py`, INTERFACE ZONE):**
```python
ANTIGEN_PDBS = {
    "6cbp": "raw_data/6cbpP_antigen_poc.pdb",
    "8x0t": "raw_data/8x0tA_antigen_poc.pdb",
    "8wsq": "raw_data/8wsqF_antigen_poc.pdb",
}

CombinedCDRs = {
    "6cbp": "YDVGSHDLVXXVNKXXXXXXXXSFGGSATVVCXXXXXXXXXXXXXXXXYDVGSHDLVXXVNKXXXXXXXXSFGGSATVVCXXXXXXXXXXXXXXX",
    "8x0t": "FTFSSFNMXXXDDDGSYPNXXXKSGPRPPHTTYWHXXXXXXXXXXXXXSWCPFCFYXXXNNKXXXXXXXXSSDFRRWAFXXXXXXXXXXXXXXXX",
    "8wsq": "FSFTNYGMXXXSYDDGSDKYXXRDPTGDYGDFPEQXXXXXXXXXXXXXLNIGSNYVXXXNNQXXXXXXXXAWDDSLSGVVFXXXXXXXXXXXXXX",
}
```

---

# Position Selection

Choose the **position strategy** in the INTERFACE ZONE:

- `"all_nonX"` — mutate all positions where residue ≠ `X`.
- `"first_n"` — mutate the first `N` non-`X` positions (`FIRST_N`).
- `"custom"` — provide explicit zero-based lists per key:
  ```python
  CUSTOM_POSITIONS = {
      "6cbp": [31, 33, 52, 57, 60, 98],
      "8x0t": [30, 32, 50, 56, 62, 95],
      "8wsq": [10, 12, 34],  # example
  }
  ```

---

# Parameters (bottom, PARAMETERS section)

- `K_MUTANTS` — Number of final variants per sequence.
- `TEMPERATURE` — Softmax temperature for amino-acid sampling (↑ = more exploratory).
- `TOP_K`, `TOP_P` — Sampling constraints (Top-K or nucleus). Use `None` to disable either.
- `AA_BLACKLIST` — Residues to forbid (default `"C"`). Set `""` to allow all 20 AAs.
- `SCHEME` — Position weighting for **which site to mutate next**:  
  `"uniform"`, `"anti_porig"`, `"entropy"`, `"margin_inverse"`, `"combo"`, `"combo3"`.
- `ALPHA`, `BETA`, `GAMMA` — Weights for `"combo"`/`"combo3"`:
  - `ALPHA` → uncertainty (normalized entropy).
  - `BETA`  → (1 − p_orig) emphasis.
  - `GAMMA` → (1 − margin) emphasis (smaller top-1 vs top-2 gap → more likely to mutate).
- `FORCE_MUTATE` — If `True`, the chosen residue must differ from the original at that position.
- `SEED` — Random seed (`None` for non-deterministic runs).

---

# Usage Examples

## Jupyter
- Paste `mutate_epi.py` content into a notebook cell, edit **INTERFACE ZONE** + **PARAMETERS**, run to get variants, logs, and summaries.

## CLI
```bash
python mutate_epi.py
```

---

# Output Format

The main entrypoints return objects of the form:
```text
Dict[str, List[Tuple[str, List[dict]]]]
```
For each antigen key:
- a list of `K_MUTANTS` variants, where each variant is:
  - `mut_seq: str` — the final mutated sequence.
  - `history: List[dict]` — per-step diagnostics including:
    - picked position, previous/new residue, change flag, fallback info
    - top-K AA proposals & probabilities
    - per-position diagnostics (entropy, margin, weights)

---

# Notes & Limitations

- Sampling is restricted to the **20 canonical amino acids**.
- Default blacklist is `"C"`; set `AA_BLACKLIST=""` to allow cysteine.
- PDBs should represent **antigen pockets** (binding vicinity). If you only have full antigens, pre-cut pockets or provide `_antigen_poc.pdb` files.
- FiLM/Graph-Transformer dimensions must match the checkpoint.
- The sampler treats `X` as **non-mutating** sentinel positions.

---

# Citation

- Paper: [link pending]
- Codebase: [link pending]

---

# License

[license pending]
