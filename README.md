# iMut-CDR / iMut-CDR-Epi (Top-Level)

This repository hosts two closely related application packages for **in‑silico antibody CDR design**:

- **`iMut-CDR/`** — sequence-only, ESM‑2–based iterative mutator (masked‑token recovery + contrast + distill).
- **`iMut-CDR-Epi/`** — target-aware variant that **conditions on an antigen pocket** via a **Graph Transformer** and **token‑level FiLM** applied to ESM‑2 hidden states.

> This top-level README covers repo layout, shared environment setup, model assets, and a quick start for both subprojects.  
> Sub-project specifics (CLI/Jupyter examples, parameter explanations) live in each folder’s `README.md`.

---

## Repo Layout

```
iMut-CDR/
├─ .git/
├─ iMut-CDR/
│  ├─ README.md
│  └─ mutate.py
├─ iMut-CDR-Epi/
│  ├─ README.md
│  ├─ mutate.py            # FiLM-conditioned application script
│  └─ raw_data/            # place antigen pocket PDBs here (e.g., *_antigen_poc.pdb)
├─ environment.yml         # conda option
└─ requirements.txt        # pip/venv option
```

---

## What’s the difference?

- **iMut-CDR (base)**: proposes CDR mutations from sequence context only, using an ESM‑2 backbone with masked‑LM style training and regularizers (contrastive alignment, token self‑distill). Inference proceeds **one site at a time**, respecting inter‑site dependencies.
- **iMut-CDR-Epi (conditional)**: same iterative mutator, but **conditioned on a 3D antigen pocket**. A light **Graph Transformer** encodes the residue‑level pocket graph; a global pocket embedding modulates ESM‑2 token features via **FiLM (γ/β)** before the LM head, biasing proposals toward **epitope compatibility**.

---

## Environment Setup (shared)

Choose one setup path for the whole repo (both subprojects use the same Python env).

### Option A — Conda

```bash
conda env create -f environment.yml
conda activate imut-cdr
```

- GPU users: ensure your NVIDIA driver/CUDA toolkit matches the `pytorch-cuda` build in `environment.yml`.
- CPU-only users: remove/adjust the CUDA lines per PyTorch docs.

### Option B — pip + venv

```bash
python -m venv .venv
# Windows PowerShell:
. .\.venv\Scripts\Activate.ps1
# (cmd.exe)   : .\.venv\Scripts\activate.bat
# (Unix/macOS): source .venv/bin/activate
pip install -r requirements.txt
```

- GPU users: install a torch wheel compatible with your CUDA version (see PyTorch docs).
- CPU-only example:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Model Assets (shared)

Both subprojects expect:

1. **ESM‑2 (650M) weights** — download from Hugging Face (`facebook/esm2_t33_650M_UR50D`) and set `local_model_dir` inside each script (or via a constant at the top).
2. **Project checkpoint (`best.pt`)**  
   - iMut‑CDR: sequence‑only checkpoint.  
   - iMut‑CDR‑Epi: FiLM‑conditioned checkpoint trained with pocket graphs.

> Place checkpoints wherever convenient (e.g., repo root or each subfolder) and update the `weights_path` variable in the scripts.

---

## Data Inputs

- **iMut‑CDR**: antibody CDR sequences (strings of one‑letter amino acids).
- **iMut‑CDR‑Epi**: **plus** an antigen pocket structure per target. Provide pocket PDBs in:
  ```
  iMut-CDR-Epi/raw_data/
    6cbpP_antigen_poc.pdb
    8x0tA_antigen_poc.pdb
    8wsqF_antigen_poc.pdb
  ```
  The Epi app caches parsed atom tables under `iMut-CDR-Epi/data/antigens/` on first run.

---

## Quick Start

### A) iMut‑CDR (sequence‑only)

1. Open `iMut-CDR/mutate.py` and set:
   - `local_model_dir = r"D:\...\facebook\esm2_t33_650M_UR50D"`
   - `weights_path = r"D:\...\best.pt"`
   - Your sequences and mutation positions (0‑based).

2. Run:
   ```bash
   cd iMut-CDR
   python mutate.py
   ```

3. Inspect console logs for per‑iteration diagnostics and final variants.  
   See `iMut-CDR/README.md` for parameter details (temperature, Top‑K/Top‑P, blacklist, etc.).

### B) iMut‑CDR‑Epi (pocket‑conditioned)

1. Place pocket PDBs under `iMut-CDR-Epi/raw_data/` (example filenames above).
2. Open `iMut-CDR-Epi/mutate.py` and set:
   - `LOCAL_MODEL_DIR`, `WEIGHTS_PATH`
   - `ANTIGEN_PDBS` and `CombinedCDRs` (edit section labeled “interface”)
   - Position strategy (`all_nonX`, `first_n`, `custom`) and related params.

3. Run:
   ```bash
   cd iMut-CDR-Epi
   python mutate.py
   ```

4. On first run for a given antigen key, pocket PDBs are cached as NPZ.  
   See `iMut-CDR-Epi/README.md` for a detailed walkthrough and parameter meanings.


---

## Typical Workflow

1. Start with iMut‑CDR to validate that your sequences and mutation positions flow end‑to‑end.
2. Move to iMut‑CDR‑Epi once **pocket PDBs** are ready.  
   Confirm that pocket parsing and NPZ caching succeed, then iterate on sampling hyperparameters (temperature, Top‑K/Top‑P) and position weighting (`combo3`, `ALPHA/BETA/GAMMA`, etc.).
3. Compare proposed variants and pick candidates for downstream structural scoring or experimental validation.

---

