# -*- coding: utf-8 -*-
"""
FiLM-conditioned iterative multi-mutation inference (inference script, three antigen-CDR examples) — UPDATED
- Pocket Graph Transformer (8 heads) with geometric RBF distance bias
- token↔pocket cross-attention (4 heads)
- Per-token FiLM via [H; C] -> (gamma, beta) in (B,L,D)
"""

import os, gzip, json, math, random, warnings, shutil, subprocess, re, csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

# ---- backend & warning settings ----
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True

# ==== Patch A: disable unstable flash-attention and use safe kernels ====
def configure_safe_attention_kernels():
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
    except Exception:
        pass

configure_safe_attention_kernels()

# =========================================================
# ==================  Interface section  ===================
# =========================================================

RAW_DIR = Path("raw_data")
ANTIGEN_PDBS: Dict[str, Path] = {
    "6cbp": RAW_DIR / "6cbpP_antigen_poc.pdb",
    "8x0t": RAW_DIR / "8x0tA_antigen_poc.pdb",
    "8wsq": RAW_DIR / "8wsqF_antigen_poc.pdb",
}

# Example combined CDR strings (X = positions allowed to mutate are filtered later by strategy)
CombinedCDRs: Dict[str, str] = {}
CombinedCDRs["6cbp"] = "YDVGSHDLVXXVNKXXXXXXXXSFGGSATVVCXXXXXXXXXXXXXXXXYDVGSHDLVXXVNKXXXXXXXXSFGGSATVVCXXXXXXXXXXXXXXX"
CombinedCDRs["8x0t"] = "FTFSSFNMXXXDDDGSYPNXXXKSGPRPPHTTYWHXXXXXXXXXXXXXSWCPFCFYXXXNNKXXXXXXXXSSDFRRWAFXXXXXXXXXXXXXXXX"
CombinedCDRs["8wsq"] = "FSFTNYGMXXXSYDDGSDKYXXRDPTGDYGDFPEQXXXXXXXXXXXXXLNIGSNYVXXXNNQXXXXXXXXAWDDSLSGVVFXXXXXXXXXXXXXX"

ANTIGEN_NPZ_DIR = Path("data/antigens")
ANTIGEN_NPZ_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_DIR = "/data/linbu/RandomMutation/models/esm2_650m"   # ← change to your local ESM2 directory
WEIGHTS_PATH    = "/data/linbu/ConditionalMutation/runs_cond_film_token_xattn_item_v3/best.pt"  # ← trained checkpoint (must match new arch)

# Position selection strategy for CDR mutation
POSITION_STRATEGY = "all_nonX"   # {"all_nonX","first_n","custom"}
FIRST_N = 8
CUSTOM_POSITIONS: Dict[str, List[int]] = {}

# Distance cutoff for CA–CA contact graph (Å)
DIST_CUTOFF = 8.0

# Pretrained amino-acid vectors (same as training script)
AA_VEC_PATH = "aa_vec_dic.npy"
aa_dict: Dict[str, np.ndarray] = np.load(AA_VEC_PATH, allow_pickle=True).item()
AA_VEC_DIM = int(next(iter(aa_dict.values())).shape[0])

# =========================================================
# ===================  Implementation  =====================
# =========================================================

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA21 = AA20 + "X"
AA2IDX = {a: i for i, a in enumerate(AA21)}
PDB_LINE_RE = re.compile(r"^(ATOM  |HETATM)")

def _pick_gpu_by_nvidia_smi(min_free_mb=2048, prefer_ids=None):
    if not torch.cuda.is_available():
        return None, None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            ).strip().splitlines()
            pairs = []
            for ln in out:
                a, b = [x.strip() for x in ln.split(",")]
                idx, free = int(a), int(b)
                pairs.append((idx, free))
            if prefer_ids is not None:
                prefer = set(prefer_ids)
                pairs = [p for p in pairs if p[0] in prefer]
                if not pairs:
                    return None, None
            return max(pairs, key=lambda x: x[1])
        except Exception:
            pass
    pairs = []
    for i in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(i)
            pairs.append((i, int(free_bytes / 1024 / 1024)))
        except Exception:
            pairs.append((i, 0))
    return max(pairs, key=lambda x: x[1]) if pairs else (None, None)

def best_cuda_device(min_free_mb=1024):
    """Pick a GPU with largest free memory; fallback to CPU if none available."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = None
    if visible:
        allowed = [int(x) for x in visible.split(",") if x.strip() != ""]
    phys_idx, free_mb = _pick_gpu_by_nvidia_smi(min_free_mb=min_free_mb, prefer_ids=allowed)
    if phys_idx is None:
        return torch.device("cpu")
    rel_idx = allowed.index(phys_idx) if allowed is not None else phys_idx
    dev = torch.device(f"cuda:{rel_idx}")
    torch.cuda.set_device(dev)
    print(f">> Picked GPU: physical={phys_idx}, visible={rel_idx}, free≈{free_mb} MiB")
    return dev

_HAS_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

class autocast_cuda:
    """Thin wrapper over torch.amp / legacy autocast for CUDA."""
    def __enter__(self):
        self.ctx = torch.amp.autocast("cuda") if _HAS_NEW_AMP else torch.cuda.amp.autocast()
        return self.ctx.__enter__()
    def __exit__(self, exc_type, exc, tb):
        return self.ctx.__exit__(exc_type, exc, tb)

# ---------- PDB -> npz ----------

def parse_pdb_atoms(pdb_path: Path):
    """Parse ATOM/HETATM lines and collect atom-level info compatible with training npz fields."""
    if not pdb_path.exists():
        return None
    xs, ys, zs = [], [], []
    elem, resn, resid, chain, atomn = [], [], [], [], []
    with open(pdb_path, "r") as f:
        for line in f:
            if not PDB_LINE_RE.match(line):
                continue
            try:
                atom_name = line[12:16].strip()
                res_name  = line[17:20].strip()
                ch        = line[21].strip() or "?"
                res_seq   = int(line[22:26].strip())
                x, y, z   = float(line[30:38]), float(line[38:46]), float(line[46:54])
                element   = line[76:78].strip() or atom_name[0]
            except Exception:
                continue
            xs.append(x); ys.append(y); zs.append(z)
            elem.append(element.upper()); resn.append(res_name.upper())
            resid.append(res_seq); chain.append(ch); atomn.append(atom_name.upper())
    if not xs:
        return None
    uniq_elems = sorted(set(elem))
    elem2id = {e: i for i, e in enumerate(uniq_elems)}
    elem_id = [elem2id[e] for e in elem]
    return {
        "pos": np.stack(
            [np.array(xs, dtype="float32"),
             np.array(ys, dtype="float32"),
             np.array(zs, dtype="float32")],
            axis=1,
        ),
        "element_id": np.array(elem_id, dtype="int16"),
        "element_vocab": np.array(uniq_elems, dtype=object),
        "res_name": np.array(resn, dtype=object),
        "res_id":   np.array(resid, dtype="int32"),
        "chain_id": np.array(chain, dtype=object),
        "atom_name":np.array(atomn, dtype=object),
    }

def ensure_antigen_npz(pdb_path: Path, npz_out: Path) -> bool:
    """Convert antigen pocket PDB to residue-wise npz if not already present."""
    if npz_out.exists():
        return True
    data = parse_pdb_atoms(pdb_path)
    if data is None:
        return False
    np.savez_compressed(npz_out, **data)
    return True

# ---------- residue graph building (AA vectors + 8Å contact) ----------

def res_vec(res_name: str, dim: int = AA_VEC_DIM):
    """
    Map residue name to pretrained amino-acid vector.
    Falls back to 'X' or zeros if not found, and resizes to AA_VEC_DIM if needed.
    """
    a = (res_name.upper()[0] if res_name else "X")
    if a in aa_dict:
        v = aa_dict[a]
    elif "X" in aa_dict:
        v = aa_dict["X"]
    else:
        v = np.zeros((dim,), dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if v.shape[0] != dim:
        v = np.resize(v, (dim,)).astype(np.float32)
    return v

def build_residue_graph_from_npz(npz_path: str, cutoff: float = DIST_CUTOFF):
    """
    Build residue-level graph from antigen npz, matching the training script:

    - Node: one node per residue, feature = pretrained AA vector (AA_VEC_DIM).
    - Edges:
        (1) sequence edges: neighbors in the same chain i <-> j
        (2) contact edges: CA–CA distance <= cutoff (default 8 Å) i <-> j

    - edge_attr: [distance, is_seq] of shape (E,2)
    """
    with np.load(npz_path, allow_pickle=True) as t:
        pos_atom = t["pos"]
        res_name = t["res_name"]
        res_id   = t["res_id"]
        chain_id = t["chain_id"]
        atom_name= t["atom_name"]

    # group atoms by (chain, residue id, residue name)
    groups = defaultdict(list)
    for i in range(len(pos_atom)):
        key = (str(chain_id[i]), int(res_id[i]), str(res_name[i]))
        groups[key].append(i)

    nodes, coords = [], []
    for (ch, rid, rname), idxs in groups.items():
        # prefer CA; fallback to mean of all atoms in residue
        ca_idx = next(
            (j for j in idxs if str(atom_name[j]).upper() in ("CA", " CΑ", " C\u0391")),
            None,
        )
        if ca_idx is not None:
            p = pos_atom[ca_idx]
        else:
            p = pos_atom[idxs].mean(axis=0)
        coords.append(p.astype(np.float32))
        nodes.append((ch, rid, rname))

    R = len(nodes)
    if R > 0:
        coords = np.stack(coords, axis=0)  # (R, 3)
        feats = np.stack([res_vec(rn) for (_, _, rn) in nodes], axis=0)  # (R, AA_VEC_DIM)
    else:
        coords = np.zeros((0, 3), dtype=np.float32)
        feats  = np.zeros((0, AA_VEC_DIM), dtype=np.float32)

    # sequence edges: adjacent residues in the same chain
    ch_rid2idx = defaultdict(dict)
    for i, (ch, rid, _) in enumerate(nodes):
        ch_rid2idx[ch][rid] = i

    src, dst, is_seq = [], [], []
    for ch, rids in ch_rid2idx.items():
        order = sorted(rids.keys())
        for a, b in zip(order[:-1], order[1:]):
            i = rids[a]; j = rids[b]
            src += [i, j]; dst += [j, i]; is_seq += [1, 1]

    # contact edges: CA–CA distance <= cutoff
    if R >= 2:
        dmat = np.sqrt(np.maximum(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1), 1e-12))
        np.fill_diagonal(dmat, np.inf)
        contact = (dmat <= float(cutoff))
        c_src, c_dst = np.where(contact)
        for i, j in zip(c_src.tolist(), c_dst.tolist()):
            src.append(i)
            dst.append(j)
            is_seq.append(0)

    # construct edge tensors
    if len(src) > 0:
        src_arr = np.array(src, dtype=np.int64)
        dst_arr = np.array(dst, dtype=np.int64)
        dist = np.linalg.norm(coords[src_arr] - coords[dst_arr], axis=1)
        edge_attr = np.stack([dist.astype(np.float32), np.array(is_seq, dtype=np.float32)], axis=1)
        edge_index = np.stack([src_arr, dst_arr], axis=0)
    else:
        edge_attr = np.zeros((0, 2), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return {
        "x": torch.from_numpy(feats),
        "pos": torch.from_numpy(coords),
        "edge_index": torch.from_numpy(edge_index),
        "edge_attr": torch.from_numpy(edge_attr),
    }

# ---------- Graph Transformer + RBF geometric bias ----------

class RBF(nn.Module):
    """Radial basis function expansion over distances."""
    def __init__(self, num_k=16, dmin=0.0, dmax=20.0):
        super().__init__()
        centers = torch.linspace(dmin, dmax, num_k)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, d):
        """
        d: (E,) distances
        return: (E, num_k) RBF-expanded features
        """
        diff = d.unsqueeze(-1) - self.centers  # (E, num_k)
        return torch.exp(-self.gamma * diff * diff)

def _make_sparse_attn_mask_with_bias(g, rbf: RBF, rbf_mlp: nn.Module, use_knn_mask=True):
    """
    Build a float attention mask of shape (N+1, N+1) to be added to attention logits:

    - default value -1e9 means "fully masked"
    - allowed positions are 0 or geometric bias:
        * CLS ↔ all tokens: 0
        * self tokens: 0
        * graph edges (i,j): RBF(distance_ij) -> MLP -> scalar bias, added to logits

    This mask is treated as an additive mask on QK^T / sqrt(d).
    """
    ei = g["edge_index"]          # (2, E)
    ea = g["edge_attr"]           # (E, 2), [:,0] = distance
    N = g["x"].size(0)
    dev = ei.device

    L = N + 1  # +1 for CLS
    # initialize as fully masked
    mask = torch.full((L, L), -1e9, dtype=torch.float32, device=dev)

    if N == 0:
        # only CLS token; allow self-attention
        mask[0, 0] = 0.0
        return mask.contiguous()

    # allow CLS ↔ all tokens without geometric bias
    mask[0, :] = 0.0
    mask[:, 0] = 0.0

    # allow self tokens without geometric bias
    idx = torch.arange(N, device=dev) + 1
    mask[idx, idx] = 0.0

    # allow graph edges and add geometric RBF bias
    if ei.numel() > 0:
        src, dst = ei[0], ei[1]            # (E,)
        d = ea[:, 0].to(dev)               # (E,) distance

        phi = rbf(d)                       # (E, num_k)
        b = rbf_mlp(phi).squeeze(-1)       # (E,)

        # simple normalization and clipping
        K = phi.size(-1)
        b = b / math.sqrt(float(K) + 1e-8)
        b = torch.clamp(b, -2.0, 2.0)

        # match dtype with mask (important for AMP)
        b = b.to(mask.dtype)

        # write bias into (i+1, j+1)
        mask[src + 1, dst + 1] = b

    if not use_knn_mask:
        # turn into full-attention: no masking, no RBF bias
        mask[:, :] = 0.0

    return mask.contiguous()

class GraphTransformerBlock(nn.Module):
    """Single Graph Transformer block with MHA and MLP, using external attention mask."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, attn_mask=None):
        # x: (B, N+1, D)
        q = k = v = self.ln1(x)
        am = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=q.dtype, device=q.device)
                am = am.masked_fill(attn_mask.to(q.device), -1e9).contiguous()
            else:
                am = attn_mask.to(dtype=q.dtype, device=q.device).contiguous()
        out, _ = self.attn(q, k, v, attn_mask=am, need_weights=False)
        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x

class PocketGraphTransformer(nn.Module):
    """
    Pocket encoder:

    - consumes residue graph g (node features from AA vectors)
    - builds sparse attention mask with 8 Å contact graph and RBF distance bias
    - runs a stack of Graph Transformer blocks
    - returns:
        cond: global conditioning vector (out_dim,)
        nodes: node representations including CLS (N+1, Dp) if return_nodes=True
    """
    def __init__(self, node_dim=AA_VEC_DIM, edge_dim=2, hidden=256, layers=4,
                 out_dim=256, num_heads=8, dropout=0.0, use_knn_mask=True,
                 rbf_num_k: int = 16):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.blocks = nn.ModuleList(
            [GraphTransformerBlock(hidden, num_heads=num_heads, dropout=dropout)
             for _ in range(layers)]
        )

        # RBF and MLP for geometric bias
        self.rbf = RBF(num_k=rbf_num_k, dmin=0.0, dmax=20.0)
        self.rbf_mlp = nn.Sequential(
            nn.Linear(rbf_num_k, rbf_num_k),
            nn.SiLU(),
            nn.Linear(rbf_num_k, 1),
        )
        with torch.no_grad():
            last = self.rbf_mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        self.num_heads = num_heads
        self.use_knn_mask = use_knn_mask
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, g: dict, return_nodes: bool = True):
        dev = self.readout[-1].weight.device
        x = g["x"].to(device=dev, dtype=torch.float32)
        if x.size(0) == 0:
            cond = torch.zeros(self.readout[-1].out_features,
                               device=dev, dtype=torch.float32)
            if return_nodes:
                return cond, torch.zeros(
                    1, 0, self.blocks[0].attn.embed_dim,
                    device=dev, dtype=torch.float32,
                )
            return cond

        ei = g["edge_index"].to(device=dev, dtype=torch.long)
        ea = g["edge_attr"].to(device=dev, dtype=torch.float32)

        h = self.node_proj(x).unsqueeze(0)      # (1, N, Dp)
        cls = self.cls.expand(1, -1, -1)        # (1, 1, Dp)
        h = torch.cat([cls, h], dim=1)          # (1, N+1, Dp)

        attn_mask = _make_sparse_attn_mask_with_bias(
            {"edge_index": ei, "edge_attr": ea, "x": g["x"].to(dev)},
            self.rbf,
            self.rbf_mlp,
            self.use_knn_mask,
        ).contiguous()                           # (N+1, N+1)

        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)      # (1, N+1, Dp)

        cls_h = h[:, 0, :]                       # (1, Dp)
        cond = self.readout(cls_h).squeeze(0)    # (C,)
        if return_nodes:
            return cond, h.squeeze(0)            # (N+1, Dp)
        return cond

# ---------- Cross-attention: token ↔ pocket nodes ----------

class TokenPocketCrossAttn(nn.Module):
    """Cross-attention from token embeddings to pocket-node representations."""
    def __init__(self, d_model: int, d_pocket: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(d_model,  d_model,  bias=False)
        self.k = nn.Linear(d_pocket, d_model,  bias=False)
        self.v = nn.Linear(d_pocket, d_model,  bias=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln_q = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor, P: torch.Tensor, key_padding_mask=None):
        # H: (B, L, D), P: (B, M, Dp), key_padding_mask: (B, M) True=pad
        Q = self.q(self.ln_q(H))
        K = self.k(P)
        V = self.v(P)
        C, _ = self.attn(Q, K, V, key_padding_mask=key_padding_mask)
        return C  # (B, L, D)

# ---------- ESM2 + FiLM model (inference-only version) ----------

def _find_token_embedding(encoder):
    """Heuristically locate the token embedding layer inside the ESM encoder."""
    for path in [("embeddings", "word_embeddings"),
                 ("embed_tokens",),
                 ("encoder", "embed_tokens")]:
        m = encoder
        ok = True
        for name in path:
            if not hasattr(m, name):
                ok = False
                break
            m = getattr(m, name)
        if ok and isinstance(m, nn.Embedding):
            return m
    return None

class ProteinMLMContrastCond_Infer(nn.Module):
    """
    Inference model:

    - ESM2 encoder + tied LM head
    - PocketGraphTransformer (8 heads, RBF distance bias, returns per-node features)
    - Token↔Pocket cross-attention (4 heads)
    - Per-token FiLM via [H; C] -> (gamma, beta) -> H_mod
    - forward(seqs, mask_pos, graphs) -> seq_logits (B, L, V)
    """
    def __init__(self, local_model_dir: str, cond_dim=256, use_knn_mask=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
        self.encoder   = EsmModel.from_pretrained(local_model_dir, local_files_only=True)
        try:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

        self.model_max_len = getattr(self.encoder.config, "max_position_embeddings", None)
        self.mask_token_str = getattr(self.tokenizer, "mask_token", None) or "<mask>"

        vocab_size = len(self.tokenizer)
        aa_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for aa in AA20:
            tid = self.tokenizer.convert_tokens_to_ids(aa)
            if isinstance(tid, int) and tid >= 0:
                aa_mask[tid] = True
        self.register_buffer("aa_mask", aa_mask, persistent=False)

        d_model = self.encoder.config.hidden_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)
        emb = _find_token_embedding(self.encoder)
        if emb is not None and emb.weight.shape == self.lm_head.weight.shape:
            self.lm_head.weight = emb.weight
            print(">> LM head tied to token embedding.")

        # Pocket encoder + per-node representations (including CLS) for cross-attention
        self.pocket = PocketGraphTransformer(
            node_dim=AA_VEC_DIM,
            edge_dim=2,
            hidden=256,
            layers=4,
            out_dim=cond_dim,
            num_heads=8,
            dropout=0.0,
            use_knn_mask=use_knn_mask,
        )

        # Token↔Pocket cross-attention (4 heads)
        self.cross = TokenPocketCrossAttn(d_model=d_model, d_pocket=256, n_heads=4, dropout=0.0)

        # Per-token FiLM via [H; C] -> (gamma, beta)
        self.film_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, 2 * d_model),
        )
        with torch.no_grad():
            last = self.film_mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            last.bias[:d_model].fill_(1.0)  # gamma ~ 1, beta ~ 0 at init

    def _apply_real_mask(self, seqs, mask_pos):
        """Replace selected positions with the ESM mask token."""
        ms = self.mask_token_str
        out = []
        for s, pos_list in zip(seqs, mask_pos):
            arr = list(s)
            for p in pos_list:
                if 0 <= p < len(arr):
                    arr[p] = ms
            out.append("".join(arr))
        return out

    def _embed(self, seq_batch):
        """Run ESM encoder to obtain per-token embeddings and attention mask."""
        with autocast_cuda():
            toks = self.tokenizer(
                seq_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_max_len,
                add_special_tokens=False,
            ).to(next(self.parameters()).device)
            out = self.encoder(**toks)
            return out.last_hidden_state, toks["attention_mask"]  # (B, L, D), (B, L)

    def _prep_pocket_batch(self, graphs: List[dict], dev: torch.device):
        """
        Run pocket encoder on a list of residue graphs and pad node matrices for cross-attention.
        Returns:
            cond: (B, C)
            P:    (B, Mmax, Dp)
            kpm:  (B, Mmax) True=pad
        """
        cond_list, node_list, lengths = [], [], []
        # we keep pocket encoder in full precision to match training
        with torch.cuda.amp.autocast(enabled=False):
            for g in graphs:
                c, nodes = self.pocket(g, return_nodes=True)   # c: (C,), nodes: (M, Dp) incl CLS
                cond_list.append(c)
                node_list.append(nodes)
                lengths.append(nodes.size(0))
        cond = torch.stack(cond_list, dim=0).to(dev)           # (B, C)
        B = len(node_list)
        Mmax = max(lengths) if lengths else 0
        Dp = node_list[0].size(-1) if Mmax > 0 else 256
        P = torch.zeros((B, Mmax, Dp), device=dev, dtype=torch.float32)
        kpm = torch.ones(B, Mmax, dtype=torch.bool, device=dev)  # True=pad
        for i, n in enumerate(node_list):
            m = n.size(0)
            if m > 0:
                P[i, :m, :] = n.to(dev, dtype=torch.float32)
                kpm[i, :m] = False
        return cond, P, kpm

    def forward(self, seqs: List[str], mask_pos: List[List[int]], graphs: List[dict]):
        dev = next(self.parameters()).device

        # 1) pocket batch: cond and node matrices
        cond, P, kpm = self._prep_pocket_batch(graphs, dev)

        # 2) ESM embeddings for masked sequences
        masked_seqs = self._apply_real_mask(seqs, mask_pos)
        H_masked, _ = self._embed(masked_seqs)        # (B, L, D)
        D = H_masked.size(-1)

        # 3) cross-attention to obtain pocket context per token
        C_masked = self.cross(H_masked, P, key_padding_mask=kpm)  # (B, L, D)

        # 4) FiLM: [H; C] -> (gamma, beta) -> H_mod
        X_masked = torch.cat([H_masked, C_masked], dim=-1)        # (B, L, 2D)
        gb_m = self.film_mlp(X_masked)                            # (B, L, 2D)
        gamma_m, beta_m = gb_m[..., :D], gb_m[..., D:]
        Hm = gamma_m * H_masked + beta_m                           # (B, L, D)

        # 5) MLM logits
        seq_logits = self.lm_head(Hm)                             # (B, L, V)
        return seq_logits

# ---------- probability helpers ----------

def _softmax_over_aa(logits_1L_V: torch.Tensor, aa_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Softmax over amino-acid vocab only, with temperature and proper renormalization.
    logits_1L_V: (L, V)
    """
    L, V = logits_1L_V.shape
    aa_mask = aa_mask.to(logits_1L_V.device)
    neg_fill = torch.finfo(logits_1L_V.dtype).min
    logits = logits_1L_V / max(temperature, 1e-8)
    logits = logits.masked_fill(~aa_mask.unsqueeze(0), neg_fill)
    probs = logits.softmax(dim=-1)
    probs = probs * aa_mask.unsqueeze(0).to(probs.dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs

def _filter_top_k_top_p(p_row: torch.Tensor, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
    """
    Apply top-k and/or top-p filtering to a probability row. Returns a renormalized distribution.
    """
    p = p_row.clone()
    if top_k is not None and 0 < top_k < p.numel():
        vals, idx = torch.topk(p, k=top_k)
        mask = torch.zeros_like(p, dtype=torch.bool)
        mask[idx] = True
        p = p.masked_fill(~mask, 0.0)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(p, descending=True)
        cumsum = torch.cumsum(sorted_p, dim=0)
        keep_mask = cumsum <= top_p
        if not keep_mask.any():
            keep_mask = torch.zeros_like(sorted_p, dtype=torch.bool)
            keep_mask[0] = True
        cutoff = sorted_p[keep_mask.nonzero().max().item()]
        p = p.masked_fill(p < cutoff, 0.0)
    p = p / p.sum().clamp_min(1e-12)
    return p

def _compute_position_weights(
    valid_positions: List[int],
    probs_all: List[torch.Tensor],
    p_orig_map: Dict[int, float],
    scheme: str = "combo3",
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-position weights based on entropy, original-AA prob, and margin.
    """
    N = len(valid_positions)
    if N == 0:
        return torch.empty(0)
    w = []
    for pos, row in zip(valid_positions, probs_all):
        p = row.detach().clone().cpu()
        p = p / p.sum().clamp_min(1e-12)
        if scheme == "uniform":
            score = 1.0
        elif scheme == "anti_porig":
            score = 1.0 - float(p_orig_map.get(pos, 0.0))
        elif scheme == "entropy":
            H = -(p * (p + 1e-12).log()).sum().item()
            score = H / math.log(max(p.numel(), 2))
        elif scheme == "margin_inverse":
            vals, _ = torch.topk(p, k=min(2, p.numel()))
            top = float(vals[0])
            second = float(vals[1] if vals.numel() > 1 else 0.0)
            score = max(0.0, 1.0 - max(0.0, top - second))
        elif scheme == "combo":
            H = -(p * (p + 1e-12).log()).sum().item()
            Hn = H / math.log(max(p.numel(), 2))
            score = (Hn ** max(alpha, 1e-6)) * ((1.0 - float(p_orig_map.get(pos, 0.0))) ** max(beta, 1e-6))
        elif scheme == "combo3":
            H = -(p * (p + 1e-12).log()).sum().item()
            Hn = H / math.log(max(p.numel(), 2))
            vals, _ = torch.topk(p, k=min(2, p.numel()))
            top = float(vals[0])
            second = float(vals[1] if vals.numel() > 1 else 0.0)
            margin_inv = max(0.0, 1.0 - max(0.0, top - second))
            score = (
                (Hn ** max(alpha, 1e-6))
                * ((1.0 - float(p_orig_map.get(pos, 0.0))) ** max(beta, 1e-6))
                * (margin_inv ** max(gamma, 1e-6))
            )
        else:
            raise ValueError(f"Unknown weight scheme: {scheme}")
        w.append(max(eps, float(score)))
    w = torch.tensor(w, dtype=torch.float32)
    s = float(w.sum().item())
    if s <= 0.0 or not torch.isfinite(w).all():
        return torch.full((N,), 1.0 / max(N, 1), dtype=torch.float32)
    return w / s

# ---------- Iterative mutator (FiLM-conditioned) ----------

class FiLMIterativeMutator:
    def __init__(self, weights_path: str, local_model_dir: str, device: Optional[str] = None):
        self.device = torch.device(device) if device else best_cuda_device(min_free_mb=1024)
        self.model = ProteinMLMContrastCond_Infer(local_model_dir=local_model_dir).to(self.device)
        # load trained weights (strict=False to tolerate additional keys)
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=self.device)
        res = self.model.load_state_dict(state, strict=False)
        if hasattr(res, "unexpected_keys") and len(res.unexpected_keys) > 0:
            print(f">> Ignored unexpected keys: {list(res.unexpected_keys)[:8]} ...")
        if hasattr(res, "missing_keys") and len(res.missing_keys) > 0:
            print(f">> Missing keys (not needed for inference): {list(res.missing_keys)[:8]} ...")
        self.model.eval()
        # Build id -> AA map for AA vocab only
        self.id2aa = {}
        aa_mask_cpu = self.model.aa_mask.bool().cpu()
        V = self.model.lm_head.out_features
        for tid in range(V):
            if aa_mask_cpu[tid]:
                tok = self.model.tokenizer.convert_ids_to_tokens(tid)
                self.id2aa[tid] = tok[0] if isinstance(tok, str) and len(tok) > 0 else None

    @torch.no_grad()
    def _batch_logits_single_mask(self, seq: str, positions: List[int], graph: dict, batch_size: int = 256):
        """Compute logits for a single sequence, masking one position at a time."""
        valid = [p for p in positions if 0 <= p < len(seq)]
        if not valid:
            return [], []
        seqs, mps, graphs = [], [], []
        for p in valid:
            seqs.append(seq)
            mps.append([int(p)])
            graphs.append(graph)
        logits_list = []
        for i in range(0, len(seqs), batch_size):
            chunk_seqs = seqs[i : i + batch_size]
            chunk_mps  = mps[i : i + batch_size]
            chunk_gs   = graphs[i : i + batch_size]
            lg = self.model(chunk_seqs, chunk_mps, chunk_gs)  # (B, L, V)
            logits_list.append(lg)
            torch.cuda.empty_cache()
        logits = torch.cat(logits_list, dim=0)
        return valid, logits

    @torch.no_grad()
    def _position_distributions(
        self,
        seq: str,
        positions: List[int],
        graph: dict,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        include_original_as_candidate: bool = False,
        aa_blacklist: str = "",
    ):
        """
        For each candidate position, compute AA probability distribution, optionally
        applying blacklist and removing the original amino acid.
        """
        valid, logits = self._batch_logits_single_mask(seq, positions, graph)
        if not valid:
            return [], [], {}
        N, L, V = logits.shape
        probs_all = []
        p_orig = {}
        aa_mask = self.model.aa_mask
        probs = _softmax_over_aa(logits.view(-1, V), aa_mask, temperature=temperature).view(N, L, V)
        for j, pos in enumerate(valid):
            row = probs[j, pos, :].clone()
            # blacklist certain amino acids
            for bad in aa_blacklist:
                tid = self.model.tokenizer.convert_tokens_to_ids(bad)
                if isinstance(tid, int) and 0 <= tid < row.numel():
                    row[tid] = 0.0
            # probability of original residue
            orig = seq[pos]
            orig_tid = self.model.tokenizer.convert_tokens_to_ids(orig)
            if isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                p_orig[pos] = float(row[orig_tid].item())
            else:
                p_orig[pos] = 0.0
            # remove original residue from candidates if requested
            if not include_original_as_candidate and isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                row[orig_tid] = 0.0
            row = _filter_top_k_top_p(row, top_k=top_k, top_p=top_p)
            # if filtered row becomes degenerate, fall back to full softmax
            if float(row.sum().item()) == 0.0:
                row = _softmax_over_aa(logits[j, pos, :].view(1, -1), aa_mask, temperature=temperature).view(-1)
                if not include_original_as_candidate and isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                    row[orig_tid] = 0.0
                if float(row.sum().item()) == 0.0:
                    row = _softmax_over_aa(logits[j, pos, :].view(1, -1), aa_mask, temperature=temperature).view(-1)
                row = row / row.sum().clamp_min(1e-12)
            probs_all.append(row)
        return valid, probs_all, p_orig

    def _topk_for_position(self, row: torch.Tensor, k: int = 5):
        vals, idx = torch.topk(row, k=min(k, row.numel()))
        return [(self.id2aa.get(int(tid), "?"), float(v)) for v, tid in zip(vals.tolist(), idx.tolist())]

    def _entropy_and_margin(self, row: torch.Tensor):
        p = row.detach().clone().cpu()
        p = p / p.sum().clamp_min(1e-12)
        H = -(p * (p + 1e-12).log()).sum().item()
        Hn = H / math.log(max(p.numel(), 2))
        vals, _ = torch.topk(p, k=min(2, p.numel()))
        top = float(vals[0])
        second = float(vals[1] if vals.numel() > 1 else 0.0)
        margin = max(0.0, top - second)
        return Hn, margin

    def _print_iter_header(self, t: int, T: float, seq: str, remaining: List[int], key: str):
        print(f"\n==== [{key}] Iter {t:02d}  (T={T:.4f})  remaining={remaining}  len(seq)={len(seq)} ====")

    def _print_pos_diag(self, valid, probs_all, p_orig, w_pos, topk=5):
        print(">> Per-position diagnostics:")
        for i, pos in enumerate(valid):
            Hn, margin = self._entropy_and_margin(probs_all[i])
            top = self._topk_for_position(probs_all[i], k=topk)
            w = float(w_pos[i].item()) if i < len(w_pos) else 0.0
            porig = float(p_orig.get(pos, 0.0))
            top_str = ", ".join([f"{aa}({p:.3f})" for aa, p in top])
            print(
                f"   - pos {pos:>4} | w={w:.4f} | p_orig={porig:.3f} | "
                f"Hn={Hn:.3f} | margin={margin:.3f} | top{topk}: {top_str}"
            )

    def _print_joint_top(self, joint_p, joint_i, valid, topn=10):
        if not joint_p:
            print(">> Joint distribution: EMPTY")
            return
        arr = list(zip(joint_p, joint_i))
        arr.sort(key=lambda x: x[0], reverse=True)
        print(f">> Joint distribution top-{min(topn, len(arr))}:")
        s = sum(joint_p) if sum(joint_p) > 0 else 1.0
        for k in range(min(topn, len(arr))):
            p, (i, tid) = arr[k]
            pos = int(valid[i])
            print(f"   #{k+1:02d}: pos={pos:>4}, aa={self.id2aa.get(int(tid),'?')}, q={p/s:.6f}")

    def _run_preamble(
        self,
        key,
        seq,
        positions,
        force_mutate,
        scheme,
        alpha,
        beta,
        gamma,
        temperature,
        top_k,
        top_p,
        aa_blacklist,
        steps,
    ):
        print("\n----- RUN SUMMARY (Before) -----")
        print(f"Key={key} | len(seq)={len(seq)}")
        print(f"Positions (0-based, {len(positions)}): {sorted(list(positions))}")
        print(f"Force-mutate (exclude original) = {force_mutate}")
        print(f"Weight scheme = {scheme} (alpha={alpha}, beta={beta}, gamma={gamma})")
        print(f"Temperature = {temperature} | Top-K={top_k} | Top-P={top_p} | AA blacklist={repr(aa_blacklist)}")
        print(f"Planned steps = {steps}")
        print("--------------------------------")

    def _run_postsummary(self, key, original_seq, final_seq, original_positions, history):
        changed_steps = []
        for h in history:
            if h["picked"]["changed"]:
                changed_steps.append(
                    (h["step"], h["picked"]["pos"], h["picked"]["prev_char"], h["picked"]["new_char"])
                )
        changed_positions_in_order = [p for (_, p, _, _) in changed_steps]
        not_changed = [p for p in original_positions if p not in changed_positions_in_order]
        print("\n----- RUN SUMMARY (After) -----")
        print(f"Key={key} | steps executed = {len(history)}")
        print(f"Actually changed = {len(changed_positions_in_order)} / requested = {len(original_positions)}")
        if changed_steps:
            print("Mutation order:")
            for step, pos, prev_c, new_c in changed_steps:
                print(f"  step {step:>2}: pos {pos:>4} {prev_c} -> {new_c}")
        else:
            print("Mutation order: <none>")
        if not_changed:
            print(f"Unchanged positions: {sorted(not_changed)}")
        print(f"Original: {original_seq}")
        print(f"Mutated : {final_seq}")
        print("--------------------------------\n")

    @torch.no_grad()
    def mutate_iterative_one(
        self,
        key: str,
        seq: str,
        positions: List[int],
        graph: dict,
        steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        include_original_as_candidate: bool = False,
        position_weight_scheme: str = "combo3",
        position_weight_alpha: float = 1.0,
        position_weight_beta: float = 1.0,
        position_weight_gamma: float = 0.7,
        aa_blacklist: str = "C",
        seed: Optional[int] = None,
        verbose: bool = True,
        print_summaries: bool = True,
    ):
        """
        Perform iterative single-position sampling with joint (position, amino-acid) distribution,
        FiLM-conditioned on the antigen pocket graph.
        """
        rng = torch.Generator(device=self.device) if seed is not None else None
        if rng is not None:
            rng.manual_seed(int(seed))
        remaining = [int(p) for p in positions if 0 <= int(p) < len(seq) and seq[int(p)] != "X"]
        steps = int(steps) if steps is not None else len(remaining)
        history = []
        cur_seq = seq

        if print_summaries:
            self._run_preamble(
                key,
                cur_seq,
                remaining,
                not include_original_as_candidate,
                position_weight_scheme,
                position_weight_alpha,
                position_weight_beta,
                position_weight_gamma,
                temperature,
                top_k,
                top_p,
                aa_blacklist,
                steps,
            )

        for t in range(1, steps + 1):
            remaining = [p for p in remaining if 0 <= p < len(cur_seq)]
            if not remaining:
                break
            if verbose:
                self._print_iter_header(t, temperature, cur_seq, remaining, key)

            valid, probs_all, p_orig = self._position_distributions(
                cur_seq,
                remaining,
                graph,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                include_original_as_candidate=include_original_as_candidate,
                aa_blacklist=aa_blacklist,
            )
            if not valid:
                if verbose:
                    print(">> No valid positions remain; stopping.")
                break

            w_pos = _compute_position_weights(
                valid,
                probs_all,
                p_orig,
                scheme=position_weight_scheme,
                alpha=position_weight_alpha,
                beta=position_weight_beta,
                gamma=position_weight_gamma,
            )
            if verbose:
                self._print_pos_diag(valid, probs_all, p_orig, w_pos, topk=5)

            joint_p, joint_i = [], []
            for i, row in enumerate(probs_all):
                row = row / row.sum().clamp_min(1e-12)
                wi = float(w_pos[i].item()) if i < len(w_pos) else 0.0
                if wi <= 0.0 or not math.isfinite(wi):
                    continue
                row = row * wi
                nz = row > 0
                if nz.any():
                    idxs = torch.nonzero(nz).view(-1).tolist()
                    for tid in idxs:
                        joint_p.append(float(row[tid].item()))
                        joint_i.append((i, tid))

            if verbose:
                self._print_joint_top(joint_p, joint_i, valid, topn=10)

            fallback_used = False
            if not joint_p:
                # fallback: sample position by w_pos and then sample AA from its row
                fallback_used = True
                wp = w_pos.to(self.device)
                if float(wp.sum().item()) == 0.0 or not torch.isfinite(wp).all():
                    wp = torch.full_like(wp, 1.0 / max(len(wp), 1))
                pos_idx = int(torch.multinomial(wp, num_samples=1, generator=rng).item())
                pos = int(valid[pos_idx])
                row = probs_all[pos_idx].clone()
                orig_tid = self.model.tokenizer.convert_tokens_to_ids(cur_seq[pos])
                if not include_original_as_candidate and isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                    row[orig_tid] = 0.0
                if float(row.sum().item()) == 0.0:
                    row2 = probs_all[pos_idx].clone()
                    if not include_original_as_candidate and isinstance(orig_tid, int) and 0 <= orig_tid < row2.numel():
                        row2[orig_tid] = 0.0
                    if float(row2.sum().item()) > 0:
                        aa_tid = int(torch.argmax(row2).item())
                    else:
                        aa_tid = int(torch.argmax(probs_all[pos_idx]).item())
                else:
                    row = row / row.sum().clamp_min(1e-12)
                    aa_tid = int(torch.multinomial(row.to(self.device), num_samples=1, generator=rng).item())
                aa_char = self.id2aa.get(int(aa_tid), None) or "A"
                picked_q = None
                pos_idx_for_diag = pos_idx
            else:
                joint = torch.tensor(joint_p, device=self.device, dtype=torch.float)
                joint = joint / joint.sum().clamp_min(1e-12)
                flat_idx = int(torch.multinomial(joint, num_samples=1, generator=rng).item())
                pos_idx, aa_tid = joint_i[flat_idx]
                pos = int(valid[pos_idx])
                aa_char = self.id2aa.get(int(aa_tid), None) or "A"
                picked_q = float(joint_p[flat_idx] / (sum(joint_p) if sum(joint_p) > 0 else 1.0))
                pos_idx_for_diag = pos_idx

            prev_char = cur_seq[pos]
            # enforce mutation if original AA was sampled again
            if (not include_original_as_candidate) and (aa_char == prev_char):
                row2 = probs_all[pos_idx_for_diag].clone()
                orig_tid = self.model.tokenizer.convert_tokens_to_ids(prev_char)
                if isinstance(orig_tid, int) and 0 <= orig_tid < row2.numel():
                    row2[orig_tid] = 0.0
                aa_tid2 = int(torch.argmax(row2).item())
                aa_char = self.id2aa.get(int(aa_tid2), aa_char)

            # collect per-position diagnostics
            pos_diag = []
            for i, p_ in enumerate(valid):
                Hn, margin = self._entropy_and_margin(probs_all[i])
                pos_diag.append(
                    {
                        "pos": int(p_),
                        "weight": float(w_pos[i].item()),
                        "p_orig": float(p_orig.get(p_, 0.0)),
                        "entropy_norm": float(Hn),
                        "margin": float(margin),
                        "topk": self._topk_for_position(probs_all[i], k=5),
                    }
                )

            if verbose:
                msg = f">> Picked: pos={pos}  {prev_char}->{aa_char}"
                if picked_q is not None:
                    msg += f"  (q≈{picked_q:.6f})"
                msg += f"  fallback={fallback_used}"
                print(msg)

            new_seq = cur_seq[:pos] + aa_char + cur_seq[pos + 1 :]
            changed = aa_char != prev_char
            if changed:
                remaining = [p for p in remaining if p != pos]

            step_info = {
                "step": t,
                "T": temperature,
                "remaining_before": [int(x) for x in valid],
                "position_diagnostics": pos_diag,
                "picked": {
                    "pos": int(pos),
                    "prev_char": prev_char,
                    "new_char": aa_char,
                    "changed": bool(changed),
                    "fallback": bool(fallback_used),
                    "q": picked_q,
                },
                "remaining_after": [int(x) for x in remaining],
            }
            if joint_p:
                arr = list(zip(joint_p, joint_i))
                arr.sort(key=lambda x: x[0], reverse=True)
                topn = min(10, len(arr))
                jt = []
                denom = sum(joint_p) if sum(joint_p) > 0 else 1.0
                for k in range(topn):
                    p_, (i_, tid_) = arr[k]
                    jt.append(
                        {"pos": int(valid[i_]), "aa": self.id2aa.get(int(tid_), "?"), "q": float(p_ / denom)}
                    )
                step_info["joint_top"] = jt
            else:
                step_info["joint_top"] = []

            history.append(step_info)
            cur_seq = new_seq
            if not remaining:
                break

        if print_summaries:
            self._run_postsummary(
                key,
                original_seq=seq,
                final_seq=cur_seq,
                original_positions=list(positions),
                history=history,
            )
        return cur_seq, history

# =========================================================
# ===============  High-level configuration  ==============
# =========================================================

K_MUTANTS            = 2
TEMPERATURE          = 1.0
TOP_K                = 8
TOP_P                = None
AA_BLACKLIST         = "C"

SCHEME               = "combo3"
ALPHA                = 1.0
BETA                 = 1.0
GAMMA                = 0.7

FORCE_MUTATE         = True
SEED                 = 0

# =========================================================
# =======================  Runner  ========================
# =========================================================

def choose_positions_for_key(key: str, seq: str) -> List[int]:
    """
    Choose candidate positions for mutation according to POSITION_STRATEGY.
    Positions with 'X' are always skipped.
    """
    if POSITION_STRATEGY == "custom":
        plist = CUSTOM_POSITIONS.get(key, [])
        return [p for p in plist if 0 <= p < len(seq) and seq[p] != "X"]
    non_x = [i for i, ch in enumerate(seq) if ch != "X"]
    if POSITION_STRATEGY == "first_n":
        return non_x[:FIRST_N]
    return non_x

def ensure_all_npz() -> Dict[str, str]:
    """Ensure each antigen pocket PDB is converted to a residue npz."""
    key2npz = {}
    for key, pdb in ANTIGEN_PDBS.items():
        out = ANTIGEN_NPZ_DIR / f"{key}.npz"
        ok = ensure_antigen_npz(pdb, out)
        if not ok:
            raise FileNotFoundError(f"[{key}] Failed to parse pocket PDB: {pdb}")
        key2npz[key] = str(out)
    return key2npz

def main_run():
    # 1) Build pocket graphs
    key2npz = ensure_all_npz()
    key2graph = {k: build_residue_graph_from_npz(pth, cutoff=DIST_CUTOFF) for k, pth in key2npz.items()}

    # 2) Initialize mutator
    mut = FiLMIterativeMutator(weights_path=WEIGHTS_PATH, local_model_dir=LOCAL_MODEL_DIR, device=None)

    # 3) Generate K mutants per key
    results: Dict[str, List[Tuple[str, list]]] = {}
    for key, seq in CombinedCDRs.items():
        if key not in key2graph:
            print(f"!! Skipping {key} (no pocket graph)")
            continue
        positions = choose_positions_for_key(key, seq)
        print("\n" + "=" * 80)
        print(f"====================  {key}  ====================")
        print("=" * 80)
        variants = []
        for kidx in range(int(K_MUTANTS)):
            print(f"\n---------- Variant {kidx+1}/{K_MUTANTS} ----------")
            new_seq, hist = mut.mutate_iterative_one(
                key=key,
                seq=seq,
                positions=positions,
                graph=key2graph[key],
                steps=len(positions),
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                include_original_as_candidate=(not FORCE_MUTATE),
                position_weight_scheme=SCHEME,
                position_weight_alpha=ALPHA,
                position_weight_beta=BETA,
                position_weight_gamma=GAMMA,
                aa_blacklist=AA_BLACKLIST,
                seed=SEED,
                verbose=True,
                print_summaries=True,
            )
            variants.append((new_seq, hist))
        results[key] = variants

    # 4) Print a brief summary
    for key, vars_ in results.items():
        print(f"\n### Key={key} | K={len(vars_)}")
        for vi, (mut_seq, hist) in enumerate(vars_):
            changed = sum(h["picked"]["changed"] for h in hist)
            order = [
                (h["picked"]["pos"], f'{h["picked"]["prev_char"]}->{h["picked"]["new_char"]}')
                for h in hist
                if h["picked"]["changed"]
            ]
            print(f"  - var#{vi+1}: changed={changed}, order={order}")
    return results

if __name__ == "__main__":
    _ = main_run()
