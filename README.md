# iMut-CDR

**iMut-CDR** (Iterative Mutator of Antibody CDRs) is a computational framework for **in-silico optimization of antibody complementarity-determining regions (CDRs)**.  
The method formulates CDR refinement as a masked-residue recovery problem. By combining cross-entropy learning with contrastive alignment and token-level self-distillation, iMut-CDR captures both local residue context and global sequence-level constraints. During inference, mutations are applied iteratively—one site at a time—so that each substitution naturally respects co-evolutionary dependencies among positions.

## Pretrained Model

A pretrained model checkpoint (**best.pt**) is available for download:

[Download best.pt](https://drive.google.com/file/d/1mLfoSNwKDw0c9Fmc1ajxK7nrHLgFSKp-/view?usp=sharing)

## Repository Scope

This repository provides the **code for applying iMut-CDR to perform in-silico mutagenesis of antibody CDRs** using the pretrained model.  
It is focused on mutation and inference workflows.

---

## Environment Setup

You can set up the environment using **Conda** (via `environment.yml`) or plain **pip** (via `requirements.txt`). Choose one of the following options.

### Option A — Conda (`environment.yml`)

1. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate imut-cdr
2. GPU users: ensure your CUDA driver is compatible with the `pytorch-cuda` version in `environment.yml`.  
   CPU-only users: remove the `pytorch-cuda` line in `environment.yml` and/or install CPU builds per PyTorch docs.

### Option B — pip (`requirements.txt`)

1. (Recommended) Use a clean virtual environment (e.g., `venv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. GPU users: install a torch build that matches your CUDA version (see official PyTorch docs).  
   CPU-only users, for example:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **Model files**: Download **ESM-2-650M** from Hugging Face  
> [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)  
> and set `local_model_dir` to the folder where those files are stored.


## Usage

> **Where is the example?**  
> The code block below is located **at the end of `mutate.py`**.  
> Please **edit the variables and function calls inside the script** (e.g., `weights_path`, `local_model_dir`, `seqs`, `positions_list`) to run your own data.

### External Input Example (embedded at the end of `mutate.py`)
```python
##### External Input Data #######
# 1) Initialization
mut = IterativeMutator(
    weights_path="best.pt",
    local_model_dir="/data/linbu/RandomMutation/models/esm2_650m",
)

# 2) External multiple sequences + their mutation positions (0-based)
seqs = [
    "QSLLGTSGKTXQVSXXXXXXXXWQGTHFPYTXXXXXXXXXXXXXXXXXGFTFNNYDXXXISYDGSSTXXXARLGHYXXXXXXXXXXXXXXXXXXX",
    "QNINKYXXXXXNTNXXXXXXXXLQHNSGWTXXXXXXXXXXXXXXXXXXGDTITAYYXXXIDPEDDSTXXXTTGVAGPYYFDYXXXXXXXXXXXXX",
    "EDIHNGXXXXXDAKXXXXXXXXQQYYDFPLTXXXXXXXXXXXXXXXXXGFTFSNYGXXXINVNSATXXXXARRSTTVPYNWFAYXXXXXXXXXXX"
]
positions_list = [
    [31, 33, 52, 57, 60, 98],  # corresponding to seqs[0]
    [30, 32, 50, 56, 62, 95],  # corresponding to seqs[1]
]

outs = run_iterative_for_many(
    mut, seqs, positions_list,
    k_mutants=3,                   # generate 3 variants per sequence
    include_original=False,        # force mutation (exclude original amino acid)
    scheme="combo3", alpha=1.0, beta=1.0, gamma=0.7,  # position weighting scheme
    temperature=1.0, top_k=8, top_p=None,
    aa_blacklist="C",              # empty string "" means no restriction
    verbose=True, print_summaries=True
)

# 3) Read results (K variants for each sequence)
for si, variants in enumerate(outs):
    print(f"\n### Sequence {si} results: K={len(variants)}")
    for vi, (mut_seq, hist) in enumerate(variants):
        changed = sum(h["picked"]["changed"] for h in hist)
        order = [ (h["picked"]["pos"], f'{h["picked"]["prev_char"]}->{h["picked"]["new_char"]}') for h in hist if h["picked"]["changed"] ]
        print(f"  - var#{vi+1}: changed={changed}, order={order}")
        # mut_seq is the final mutated sequence
```

### Argument Reference

#### Initialization (`IterativeMutator`)
- **`weights_path`** — Path to the pretrained checkpoint file (e.g., `best.pt`).
- **`local_model_dir`** — Local directory containing the **ESM-2-650M** model files.  
  The model can be downloaded from [Hugging Face](https://huggingface.co/facebook/esm2_t33_650M_UR50D), and this path should point to the folder where the files are stored.

#### `run_iterative_for_many(...)`
- **`seqs`** — A list of antibody amino-acid sequences (uppercase single-letter codes, length 129).  
  Each item corresponds to one sequence string.
- **`positions_list`** — List of integer lists (one per sequence), containing **0-based indices** into the corresponding string in `seqs[i]`.
  - Example: index `0` is the first residue of the sequence string.
  - Ensure every index is valid for its corresponding sequence.
- **`k_mutants`** — Number of **final variants per input sequence**.
- **`include_original`** — If `False`, enforce a true mutation (the selected residue must differ from the original at that position).
- **`scheme`, `alpha`, `beta`, `gamma`** — Position-weighting scheme and hyperparameters (e.g., `"combo3"`). These control site prioritization during the iterative process.
- **`temperature`** — Sampling temperature for the amino-acid distribution (higher ⇒ more exploratory).
- **`top_k`**, **`top_p`** — Optional sampling constraints.
  - `top_k=8` restricts choices to the top-8 most probable residues.
  - `top_p` (nucleus sampling) restricts to the smallest set of residues whose cumulative probability ≥ `p` (use `None` to disable).
- **`aa_blacklist`** — String of residues to **forbid** (e.g., `"C"` to avoid cysteines; `""` means no restriction).
- **`verbose`**, **`print_summaries`** — If `True`, print per-iteration diagnostics and concise run summaries.

#### Output: `outs`
- **Type**: `List[List[Tuple[str, List[dict]]]]`
  - **Outer list** — One item per input sequence.
  - **Inner list** — `k_mutants` variants per sequence.
  - **Each variant** — A tuple of:
    - **`mut_seq`** — Final mutated sequence (string).
    - **`hist`** — List of per-step dictionaries describing the iterative mutation process. Typical keys include:
      - `picked.pos` (0-based position),
      - `picked.prev_char` / `picked.new_char`,
      - `picked.changed` (bool),
      - plus optional scores/probabilities depending on configuration.

#### How to Run
```bash
python mutate.py
