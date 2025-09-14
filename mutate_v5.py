# -*- coding: utf-8 -*-
"""
Iterative multi-mutation inference (v4, verbose logging + run summaries)
-----------------------------------------------------------------------
- 匹配 train_v4.py 的 tokenizer/遮罩约定
- 迭代式多突变：一次只改一个位点，直到改完
- 两套接口：
  A) 传入序列与位点（单条/批量/每条K个变体）
  B) 从 jsonl(.gz)/json 数据集抽样，再各自产生 K 个变体
- 位置权重 scheme：{uniform, anti_porig, entropy, margin_inverse, combo, combo3}
  * combo3 = 熵^alpha × (1-p_orig)^beta × (1-margin)^gamma
- 详细日志：每轮打印并写入 history（位置诊断、Top-K、联合分布 Top-N、采样结果/兜底路径）
- 新增：每次迭代前/后“总览总结”；序列之间打印明显分割符

Author: Linbu + ChatGPT
"""

import os, gzip, json, random, math, warnings, subprocess, shutil
from typing import List, Tuple, Dict, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True

# ========== Device helpers ==========
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
                    raise RuntimeError(f"无可用GPU（受限于 CUDA_VISIBLE_DEVICES={prefer_ids}）")
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

def best_cuda_device(min_free_mb=2048):
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

# ========== AMP (兼容新旧 API) ==========
_HAS_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

class autocast_cuda:
    def __enter__(self):
        self.ctx = torch.amp.autocast("cuda") if _HAS_NEW_AMP else torch.cuda.amp.autocast()
        return self.ctx.__enter__()
    def __exit__(self, exc_type, exc, tb):
        return self.ctx.__exit__(exc_type, exc, tb)

# ========== 默认路径（可在构造器传参覆盖）==========
LOCAL_MODEL_DIR = "/data/linbu/RandomMutation/models/esm2_650m"
WEIGHTS_PATH    = "runs_esm2_fast_contrast_v5/best.pt"

# ========== Model（与训练一致的必要子集）==========
class ProteinMLMContrastModel(nn.Module):
    def __init__(self, local_model_dir: str = LOCAL_MODEL_DIR):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
        self.encoder   = EsmModel.from_pretrained(local_model_dir, local_files_only=True)
        try:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass

        self.model_max_len = getattr(self.encoder.config, "max_position_embeddings", None)
        self.mask_token_str = getattr(self.tokenizer, "mask_token", None) or "<mask>"
        print(f">> Mask token in use: {self.mask_token_str!r}")

        # 仅 20AA
        aa_tokens = list("ACDEFGHIKLMNPQRSTVWY")
        vocab_size = len(self.tokenizer)
        aa_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for aa in aa_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(aa)
            if isinstance(tid, int) and tid >= 0:
                aa_mask[tid] = True
        self.register_buffer("aa_mask", aa_mask, persistent=False)

        d_model = self.encoder.config.hidden_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

        # 绑权重（若形状一致）
        emb = self._find_token_embedding(self.encoder)
        if emb is not None and emb.weight.shape == self.lm_head.weight.shape:
            self.lm_head.weight = emb.weight
            print(">> LM head tied to token embedding weights.")

    @staticmethod
    def _find_token_embedding(encoder):
        candidates = [("embeddings", "word_embeddings"), ("embed_tokens",), ("encoder", "embed_tokens")]
        for path in candidates:
            m = encoder; ok = True
            for name in path:
                if not hasattr(m, name):
                    ok = False; break
                m = getattr(m, name)
            if ok and isinstance(m, nn.Embedding):
                return m
        return None

    def _embed(self, seq_batch: List[str]):
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
            return out.last_hidden_state, toks["attention_mask"]

    def _apply_real_mask(self, seqs: List[str], mask_pos: List[List[int]]):
        ms = self.mask_token_str
        masked = []
        for s, pos_list in zip(seqs, mask_pos):
            s_list = list(s)
            for p in pos_list:
                if 0 <= p < len(s_list):
                    s_list[p] = ms
            masked.append("".join(s_list))
        return masked

    def forward(self, seqs: List[str], mask_pos: List[List[int]]):
        masked_seqs = self._apply_real_mask(seqs, mask_pos)
        H_masked, _ = self._embed(masked_seqs)
        seq_logits = self.lm_head(H_masked)  # [B,L,V]
        return seq_logits

# ========== 概率工具（仅 20AA）==========
def _softmax_over_aa(logits_1L_V: torch.Tensor, aa_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
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
    p = p_row.clone()
    if top_k is not None and 0 < top_k < p.numel():
        vals, idx = torch.topk(p, k=top_k)
        mask = torch.zeros_like(p, dtype=torch.bool); mask[idx] = True
        p = p.masked_fill(~mask, 0.0)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(p, descending=True)
        cumsum = torch.cumsum(sorted_p, dim=0)
        keep_mask = cumsum <= top_p
        if not keep_mask.any():
            keep_mask = torch.zeros_like(sorted_p, dtype=torch.bool); keep_mask[0] = True
        cutoff = sorted_p[keep_mask.nonzero().max().item()]
        p = p.masked_fill(p < cutoff, 0.0)
    p = p / p.sum().clamp_min(1e-12)
    return p

# ========== 位置权重（含地板 eps & 全零回退）==========
def _compute_position_weights(valid_positions: List[int],
                              probs_all: List[torch.Tensor],
                              p_orig_map: Dict[int, float],
                              scheme: str = "uniform",
                              alpha: float = 1.0,
                              beta: float = 1.0,
                              gamma: float = 1.0,
                              eps: float = 1e-6) -> torch.Tensor:
    """
    返回 shape [N] 的位置权重，和为 1。
    为防 combo3 在极端情况下权重全 0，加地板 eps，并在总和仍为 0 时回退为均匀分布。
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
            top = float(vals[0]); second = float(vals[1] if vals.numel() > 1 else 0.0)
            margin = max(0.0, top - second)
            score = max(0.0, 1.0 - margin)
        elif scheme == "combo":
            H = -(p * (p + 1e-12).log()).sum().item(); Hn = H / math.log(max(p.numel(), 2))
            base = (Hn ** max(alpha,1e-6)) * ((1.0 - float(p_orig_map.get(pos, 0.0))) ** max(beta,1e-6))
            score = base ** max(gamma, 1e-6)
        elif scheme == "combo3":
            H = -(p * (p + 1e-12).log()).sum().item(); Hn = H / math.log(max(p.numel(), 2))
            vals, _ = torch.topk(p, k=min(2, p.numel()))
            top = float(vals[0]); second = float(vals[1] if vals.numel() > 1 else 0.0)
            margin = max(0.0, top - second)
            margin_inv = max(0.0, 1.0 - margin)
            score = (Hn ** max(alpha,1e-6)) * ((1.0 - float(p_orig_map.get(pos, 0.0))) ** max(beta,1e-6)) * (margin_inv ** max(gamma,1e-6))
        else:
            raise ValueError(f"Unknown weight scheme: {scheme}")
        w.append(max(eps, float(score)))
    w = torch.tensor(w, dtype=torch.float32)
    s = float(w.sum().item())
    if s <= 0.0 or not torch.isfinite(w).all():
        w = torch.full((N,), 1.0 / max(N, 1), dtype=torch.float32)
    else:
        w = w / s
    return w

# ========== 迭代突变器（详日志 + 前后总结）==========
class IterativeMutator:
    """
    用法：
        mut = IterativeMutator(weights_path="runs_esm2_fast_contrast_v5/best.pt")
        new_seq, hist = mut.mutate_iterative(seq, positions=[...], include_original_as_candidate=False, ...)
    """

    def __init__(self, device: Optional[str] = None, weights_path: str = WEIGHTS_PATH, local_model_dir: str = LOCAL_MODEL_DIR):
        self.device = torch.device(device) if device else best_cuda_device(min_free_mb=1024)
        self.model = ProteinMLMContrastModel(local_model_dir=local_model_dir).to(self.device)
        # 加载权重（忽略训练时的投影头等）
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=self.device)
        res = self.model.load_state_dict(state, strict=False)
        if len(res.unexpected_keys) > 0:
            print(f">> Ignored unexpected keys (not needed for inference): {list(res.unexpected_keys)}")
        if len(res.missing_keys) > 0:
            print(f">> Missing keys (not in ckpt; fine for inference): {list(res.missing_keys)}")
        self.model.eval()
        self.id2aa = self._build_id2aa()

    def _build_id2aa(self) -> Dict[int, str]:
        id2aa = {}
        aa_mask_cpu = self.model.aa_mask.bool().cpu()
        V = self.model.lm_head.out_features
        for tid in range(V):
            if aa_mask_cpu[tid]:
                tok = self.model.tokenizer.convert_ids_to_tokens(tid)
                id2aa[tid] = tok[0] if isinstance(tok, str) and len(tok) > 0 else None
        return id2aa

    # ----------- Pretty printing helpers (new) -----------
    def _print_run_preamble(self, seq: str, positions: List[int],
                            include_original_as_candidate: bool,
                            position_weight_scheme: str,
                            position_weight_alpha: float,
                            position_weight_beta: float,
                            position_weight_gamma: float,
                            temperature: float,
                            temperature_schedule: Optional[Tuple[float, float]],
                            top_k: Optional[int],
                            top_p: Optional[float],
                            aa_blacklist: str,
                            steps: int):
        print("\n----- RUN SUMMARY (Before) -----")
        print(f"Sequence length = {len(seq)}")
        print(f"Positions to mutate (0-based, {len(positions)}): {sorted(list(positions))}")
        print(f"Force-mutate (exclude original aa) = {not include_original_as_candidate}")
        print(f"Weight scheme = {position_weight_scheme} (alpha={position_weight_alpha}, beta={position_weight_beta}, gamma={position_weight_gamma})")
        if temperature_schedule is None:
            print(f"Temperature = {temperature}")
        else:
            t0, t1 = temperature_schedule
            print(f"Temperature schedule: {t0} -> {t1} (linear over {steps} steps)")
        print(f"Top-K = {top_k}, Top-P = {top_p}, AA blacklist = {repr(aa_blacklist)}")
        print(f"Planned steps = {steps}")
        print("--------------------------------")

    def _print_run_postsummary(self, original_seq: str, final_seq: str, original_positions: List[int], history: List[Dict]):
        changed_steps = []
        for h in history:
            if h["picked"]["changed"]:
                changed_steps.append((h["step"], h["picked"]["pos"], h["picked"]["prev_char"], h["picked"]["new_char"]))
        changed_positions_in_order = [p for (_, p, _, _) in changed_steps]
        not_changed = [p for p in original_positions if p not in changed_positions_in_order]

        print("\n----- RUN SUMMARY (After) -----")
        print(f"Total steps executed = {len(history)}")
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

    # ----------- Core ops -----------
    @torch.no_grad()
    def _batch_logits_single_mask(self, seq: str, positions: List[int], batch_size: int = 256):
        """对每个位置单独 mask，批量前向，返回 (valid_positions, logits[N,L,V])"""
        valid = [p for p in positions if 0 <= p < len(seq)]
        if not valid:
            return [], []
        seqs, mps = [], []
        for p in valid:
            seqs.append(seq); mps.append([int(p)])
        logits_list = []
        for i in range(0, len(seqs), batch_size):
            chunk_seqs = seqs[i:i+batch_size]
            chunk_mps  = mps[i:i+batch_size]
            with autocast_cuda():
                lg = self.model(chunk_seqs, chunk_mps)  # [B,L,V]
            logits_list.append(lg)
            torch.cuda.empty_cache()
        logits = torch.cat(logits_list, dim=0)
        return valid, logits

    @torch.no_grad()
    def _position_distributions(self,
                                seq: str,
                                positions: List[int],
                                temperature: float = 1.0,
                                top_k: Optional[int] = None,
                                top_p: Optional[float] = None,
                                include_original_as_candidate: bool = False,
                                aa_blacklist: str = "") -> Tuple[List[int], List[torch.Tensor], Dict[int, float]]:
        """返回 (valid_positions, [p_i(a)], p_orig map)。内部已对 20AA 归一化。"""
        valid, logits = self._batch_logits_single_mask(seq, positions)
        if not valid:
            return [], [], {}
        N, L, V = logits.shape
        probs_all = []
        p_orig = {}
        aa_mask = self.model.aa_mask
        probs = _softmax_over_aa(logits.view(-1, V), aa_mask, temperature=temperature).view(N, L, V)
        for j, pos in enumerate(valid):
            row = probs[j, pos, :].clone()  # [V]
            # ban 黑名单氨基酸
            for bad in aa_blacklist:
                tid = self.model.tokenizer.convert_tokens_to_ids(bad)
                if isinstance(tid, int) and 0 <= tid < row.numel():
                    row[tid] = 0.0
            # 原氨基酸概率 + 可选排除
            orig = seq[pos]
            orig_tid = self.model.tokenizer.convert_tokens_to_ids(orig)
            p_orig[pos] = float(row[orig_tid].item()) if (isinstance(orig_tid, int) and 0 <= orig_tid < row.numel()) else 0.0
            if not include_original_as_candidate and isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                row[orig_tid] = 0.0
            # top-k / top-p 裁剪
            row = _filter_top_k_top_p(row, top_k=top_k, top_p=top_p)
            # 兜底：若被裁剪/ban 后全为 0，放宽限制
            if float(row.sum().item()) == 0.0:
                # 撤销 topk/topp
                row = _softmax_over_aa(logits[j, pos, :].view(1, -1), aa_mask, temperature=temperature).view(-1)
                if not include_original_as_candidate and isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                    row[orig_tid] = 0.0
                if float(row.sum().item()) == 0.0:
                    # 再撤销 ban
                    row = _softmax_over_aa(logits[j, pos, :].view(1, -1), aa_mask, temperature=temperature).view(-1)
                row = row / row.sum().clamp_min(1e-12)
            probs_all.append(row)
        return valid, probs_all, p_orig

    def _topk_for_position(self, row: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        vals, idx = torch.topk(row, k=min(k, row.numel()))
        out = []
        for v, tid in zip(vals.tolist(), idx.tolist()):
            out.append((self.id2aa.get(int(tid), "?"), float(v)))
        return out

    def _entropy_and_margin(self, row: torch.Tensor) -> Tuple[float, float]:
        p = row.detach().clone().cpu()
        p = p / p.sum().clamp_min(1e-12)
        H = -(p * (p + 1e-12).log()).sum().item()
        Hn = H / math.log(max(p.numel(), 2))
        vals, _ = torch.topk(p, k=min(2, p.numel()))
        top = float(vals[0]); second = float(vals[1] if vals.numel() > 1 else 0.0)
        margin = max(0.0, top - second)
        return Hn, margin

    def _print_iter_header(self, t: int, T: float, seq: str, remaining: List[int]):
        print(f"\n==== Iter {t:02d}  (T={T:.4f})  remaining={remaining}  len(seq)={len(seq)} ====")

    def _print_pos_diag(self, valid: List[int], probs_all: List[torch.Tensor], p_orig: Dict[int, float], w_pos: torch.Tensor, topk: int = 5):
        print(">> Per-position diagnostics:")
        for i, pos in enumerate(valid):
            Hn, margin = self._entropy_and_margin(probs_all[i])
            top = self._topk_for_position(probs_all[i], k=topk)
            w = float(w_pos[i].item()) if i < len(w_pos) else 0.0
            porig = float(p_orig.get(pos, 0.0))
            top_str = ", ".join([f"{aa}({p:.3f})" for aa,p in top])
            print(f"   - pos {pos:>4} | w={w:.4f} | p_orig={porig:.3f} | Hn={Hn:.3f} | margin={margin:.3f} | top{topk}: {top_str}")

    def _print_joint_top(self, joint_p: List[float], joint_i: List[Tuple[int,int]], valid: List[int], topn: int = 10):
        if not joint_p:
            print(">> Joint distribution: EMPTY")
            return
        arr = list(zip(joint_p, joint_i))
        arr.sort(key=lambda x: x[0], reverse=True)
        print(f">> Joint distribution top-{min(topn, len(arr))}:")
        for k in range(min(topn, len(arr))):
            p, (i, tid) = arr[k]
            pos = int(valid[i])
            print(f"   #{k+1:02d}: pos={pos:>4}, aa={self.id2aa.get(int(tid),'?')}, q={p:.6f}")

    # ----------- 主流程 -----------
    @torch.no_grad()
    def mutate_iterative(self,
                         seq: str,
                         positions: List[int],
                         steps: Optional[int] = None,
                         temperature: float = 1.0,
                         top_k: Optional[int] = None,
                         top_p: Optional[float] = None,
                         include_original_as_candidate: bool = False,
                         # 别名（可选，兼容封装器/老脚本）：
                         include_original: Optional[bool] = None,
                         scheme: Optional[str] = None,
                         alpha: Optional[float] = None,
                         beta: Optional[float] = None,
                         gamma: Optional[float] = None,
                         # 真正生效的权重控制项：
                         position_weight_scheme: str = "uniform",
                         position_weight_gamma: float = 1.0,
                         position_weight_alpha: float = 1.0,
                         position_weight_beta: float = 1.0,
                         aa_blacklist: str = "C",
                         temperature_schedule: Optional[Tuple[float, float]] = None,
                         seed: Optional[int] = None,
                         verbose: bool = True,
                         print_summaries: bool = True) -> Tuple[str, List[Dict]]:
        """
        迭代进行位点采样并替换，直到所有 positions 被替换或达到 steps。
        返回：new_seq, history(list of dict，包含全部中间值，便于复盘)
        """
        # --- 参数别名合并 ---
        if include_original is not None:
            include_original_as_candidate = bool(include_original)
        if scheme is not None:
            position_weight_scheme = scheme
        if alpha is not None:
            position_weight_alpha = float(alpha)
        if beta is not None:
            position_weight_beta = float(beta)
        if gamma is not None:
            position_weight_gamma = float(gamma)

        rng = torch.Generator(device=self.device) if seed is not None else None
        if rng is not None:
            rng.manual_seed(int(seed))
        remaining = [int(p) for p in positions if 0 <= int(p) < len(seq)]
        steps = int(steps) if steps is not None else len(remaining)
        history = []
        cur_seq = seq

        if print_summaries:
            self._print_run_preamble(
                seq=cur_seq, positions=remaining,
                include_original_as_candidate=include_original_as_candidate,
                position_weight_scheme=position_weight_scheme,
                position_weight_alpha=position_weight_alpha,
                position_weight_beta=position_weight_beta,
                position_weight_gamma=position_weight_gamma,
                temperature=temperature, temperature_schedule=temperature_schedule,
                top_k=top_k, top_p=top_p, aa_blacklist=aa_blacklist, steps=steps
            )

        def _cur_T(t_idx: int):
            if temperature_schedule is None:
                return temperature
            t0, t1 = temperature_schedule
            T = t0 + (t1 - t0) * (t_idx / max(steps - 1, 1))
            return max(1e-4, float(T))

        for t in range(1, steps + 1):
            remaining = [p for p in remaining if 0 <= p < len(cur_seq)]
            if not remaining:
                break
            T_t = _cur_T(t-1)
            if verbose:
                self._print_iter_header(t, T_t, cur_seq, remaining)

            valid, probs_all, p_orig = self._position_distributions(
                cur_seq, remaining, temperature=T_t, top_k=top_k, top_p=top_p,
                include_original_as_candidate=include_original_as_candidate,
                aa_blacklist=aa_blacklist,
            )
            if not valid:
                if verbose: print(">> No valid positions remain (after filtering); stop.")
                break

            # 位置权重
            w_pos = _compute_position_weights(
                valid, probs_all, p_orig,
                scheme=position_weight_scheme,
                alpha=position_weight_alpha, beta=position_weight_beta, gamma=position_weight_gamma
            )
            if verbose:
                self._print_pos_diag(valid, probs_all, p_orig, w_pos, topk=5)

            # 联合分布 q(i,a) ∝ w_i * p_i(a)
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

            # 打印联合分布 Top-N
            if verbose:
                self._print_joint_top(joint_p, joint_i, valid, topn=10)

            fallback_used = False
            # 采样 (pos, aa)
            if not joint_p:
                fallback_used = True
                if verbose:
                    print(">> Joint distribution empty -> fallback to per-position sampling")
                # 位置分布兜底
                wp = w_pos.to(self.device)
                if float(wp.sum().item()) == 0.0 or not torch.isfinite(wp).all():
                    wp = torch.full_like(wp, 1.0 / max(len(wp), 1))
                pos_idx = int(torch.multinomial(wp, num_samples=1, generator=rng).item())
                pos = int(valid[pos_idx])

                # 该位置的氨基酸分布兜底
                row = probs_all[pos_idx].clone()
                orig_tid = self.model.tokenizer.convert_tokens_to_ids(cur_seq[pos])
                if not include_original_as_candidate and isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                    row[orig_tid] = 0.0
                if float(row.sum().item()) == 0.0:
                    # 仍为 0：排原后的 argmax；再不行就均匀
                    if not include_original_as_candidate and isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                        row2 = probs_all[pos_idx].clone()
                        row2[orig_tid] = 0.0
                        if float(row2.sum().item()) > 0.0:
                            aa_tid = int(torch.argmax(row2).item())
                        else:
                            row = torch.ones_like(row)
                            if isinstance(orig_tid, int) and 0 <= orig_tid < row.numel():
                                row[orig_tid] = 0.0
                            row = row / row.sum().clamp_min(1e-12)
                            aa_tid = int(torch.multinomial(row.to(self.device), num_samples=1, generator=rng).item())
                    else:
                        aa_tid = int(torch.argmax(probs_all[pos_idx]).item())
                else:
                    row = row / row.sum().clamp_min(1e-12)
                    aa_tid = int(torch.multinomial(row.to(self.device), num_samples=1, generator=rng).item())

                aa_char = self.id2aa.get(int(aa_tid), None) or "A"
                picked_q = None  # 兜底路径没有明确 q
            else:
                joint = torch.tensor(joint_p, device=self.device, dtype=torch.float)
                joint = joint / joint.sum().clamp_min(1e-12)
                flat_idx = int(torch.multinomial(joint, num_samples=1, generator=rng).item())
                pos_idx, aa_tid = joint_i[flat_idx]
                pos = int(valid[pos_idx])
                aa_char = self.id2aa.get(int(aa_tid), None) or "A"
                picked_q = float(joint_p[flat_idx] / (sum(joint_p) if sum(joint_p) > 0 else 1.0))

            prev_char = cur_seq[pos]
            # 强制突变保护
            if (not include_original_as_candidate) and (aa_char == prev_char):
                row2 = probs_all[pos_idx].clone()
                orig_tid = self.model.tokenizer.convert_tokens_to_ids(prev_char)
                if isinstance(orig_tid, int) and 0 <= orig_tid < row2.numel():
                    row2[orig_tid] = 0.0
                aa_tid2 = int(torch.argmax(row2).item())
                aa_char = self.id2aa.get(int(aa_tid2), aa_char)

            # 组装 step 诊断
            pos_diag = []
            for i, p_ in enumerate(valid):
                Hn, margin = self._entropy_and_margin(probs_all[i])
                pos_diag.append({
                    "pos": int(p_),
                    "weight": float(w_pos[i].item()),
                    "p_orig": float(p_orig.get(p_, 0.0)),
                    "entropy_norm": float(Hn),
                    "margin": float(margin),
                    "topk": self._topk_for_position(probs_all[i], k=5),
                })

            # 打印本轮选择
            if verbose:
                if picked_q is not None:
                    print(f">> Picked: pos={pos}  {prev_char}->{aa_char}  (q≈{picked_q:.6f})  fallback={fallback_used}")
                else:
                    print(f">> Picked (fallback): pos={pos}  {prev_char}->{aa_char}")

            # 应用变更
            new_seq = cur_seq[:pos] + aa_char + cur_seq[pos+1:]
            changed = (aa_char != prev_char)
            if changed:
                remaining = [p for p in remaining if p != pos]

            # 写入 history（包含本轮全部中间值）
            step_info = {
                "step": t,
                "T": T_t,
                "remaining_before": [int(x) for x in valid],
                "position_diagnostics": pos_diag,  # 含 w_i, p_orig, 熵、margin、top5
                "joint_top": [],                   # 前 10 个联合候选，便于复盘
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
            # 填入联合分布 top-10
            if joint_p:
                arr = list(zip(joint_p, joint_i))
                arr.sort(key=lambda x: x[0], reverse=True)
                topn = min(10, len(arr))
                jt = []
                denom = sum(joint_p) if sum(joint_p) > 0 else 1.0
                for k in range(topn):
                    p_, (i_, tid_) = arr[k]
                    jt.append({
                        "pos": int(valid[i_]),
                        "aa": self.id2aa.get(int(tid_), "?"),
                        "q": float(p_ / denom)
                    })
                step_info["joint_top"] = jt

            history.append(step_info)
            cur_seq = new_seq
            if not remaining:
                break

        if print_summaries:
            self._print_run_postsummary(original_seq=seq, final_seq=cur_seq, original_positions=list(positions), history=history)

        return cur_seq, history

    @torch.no_grad()
    def mutate_iterative_K(self, seq: str, positions: List[int], K: int = 3, **kwargs):
        """对单条 (seq, positions) 生成 K 个变体（K 次独立迭代采样）。"""
        outs = []
        for ki in range(int(K)):
            print(f"\n---------- Variant {ki+1}/{K} ----------")
            new_seq, hist = self.mutate_iterative(seq, positions, **kwargs)
            outs.append((new_seq, hist))
        return outs

    @torch.no_grad()
    def mutate_iterative_batch(self, seqs: List[str], positions_list: List[List[int]], K: int = 3, **kwargs):
        """对多条序列分别生成 K 个变体。返回：List[List[(new_seq, hist)]]"""
        assert len(seqs) == len(positions_list), "seqs 与 positions_list 长度不一致"
        all_out = []
        for idx, (s, pos) in enumerate(zip(seqs, positions_list)):
            print("\n" + "="*80)
            print(f"====================  SEQUENCE {idx+1}/{len(seqs)}  ====================")
            print("="*80)
            all_out.append(self.mutate_iterative_K(s, pos, K=K, **kwargs))
        print("\n" + "="*80 + "\n")
        return all_out

    @torch.no_grad()
    def topk_table(self, seq: str, positions: List[int], k: int = 5, temperature: float = 1.0):
        valid, probs_all, _ = self._position_distributions(seq, positions, temperature=temperature)
        table = {}
        for pos, row in zip(valid, probs_all):
            vals, idx = torch.topk(row, k=min(k, row.numel()))
            table[pos] = [(self.id2aa.get(int(tid), "?"), float(v)) for v, tid in zip(vals.tolist(), idx.tolist())]
        return table

# ========== I/O & 封装接口（Jupyter 友好）==========
def _load_jsonl_or_json(path: str, max_items: Optional[int] = None):
    arr = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
            for line in f:
                arr.append(json.loads(line))
                if max_items is not None and len(arr) >= max_items:
                    break
        else:  # .json
            data = json.load(f)
            if isinstance(data, list):
                arr = data[:max_items] if max_items else data
            else:
                arr = [data]
    return arr

def run_iterative_for_one(
    mutator: IterativeMutator,
    seq: str,
    positions: List[int],
    k_mutants: int = 1,
    *,
    include_original: bool = False,                     # False = 强制突变
    scheme: str = "combo3", alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.7,
    temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
    aa_blacklist: str = "C",
    temperature_schedule: Optional[Tuple[float,float]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    print_summaries: bool = True,
):
    """对单条序列执行迭代式多位点突变，返回 K 个 (mutated_seq, history)。positions 为 0-based。"""
    pos = [int(p) for p in positions if 0 <= int(p) < len(seq) and seq[int(p)] != 'X']
    if not pos:
        return [(seq, [])] * int(k_mutants)
    print("\n" + "="*80)
    print("===============  SINGLE SEQUENCE RUN  ===============")
    print("="*80)
    return mutator.mutate_iterative_K(
        seq, pos, K=int(k_mutants),
        temperature=temperature, top_k=top_k, top_p=top_p,
        include_original_as_candidate=bool(include_original),  # 映射
        position_weight_scheme=scheme,
        position_weight_gamma=float(gamma),
        position_weight_alpha=float(alpha),
        position_weight_beta=float(beta),
        aa_blacklist=aa_blacklist or "",
        temperature_schedule=temperature_schedule,
        seed=seed, verbose=verbose,
        print_summaries=print_summaries,
    )

def run_iterative_for_many(
    mutator: IterativeMutator,
    seqs: List[str],
    positions_list: List[List[int]],
    k_mutants: int = 1,
    **kwargs
):
    """批量版本：seqs 与 positions_list 一一对应；返回 List[List[(mut_seq, hist)]]."""
    assert len(seqs) == len(positions_list), "seqs 与 positions_list 长度不一致"
    clean_pos_list = []
    for s, pos in zip(seqs, positions_list):
        pos = [int(p) for p in pos if 0 <= int(p) < len(s) and s[int(p)] != 'X']
        clean_pos_list.append(pos)

    # 关键映射（兼容封装层命名）
    kwargs2 = dict(kwargs)
    if "include_original" in kwargs2:
        kwargs2["include_original_as_candidate"] = kwargs2.pop("include_original")
    if "scheme" in kwargs2:
        kwargs2["position_weight_scheme"] = kwargs2.pop("scheme")
    if "alpha" in kwargs2:
        kwargs2["position_weight_alpha"] = kwargs2.pop("alpha")
    if "beta" in kwargs2:
        kwargs2["position_weight_beta"] = kwargs2.pop("beta")
    if "gamma" in kwargs2:
        kwargs2["position_weight_gamma"] = kwargs2.pop("gamma")

    return mutator.mutate_iterative_batch(seqs, clean_pos_list, K=int(k_mutants), **kwargs2)

def sample_from_dataset(
    train_path: Optional[str],
    val_path: Optional[str],
    n_from_train: int = 0,
    n_from_val: int = 10,
    n_mut_each: int = 6,
    seed: int = 0,
):
    """从数据集随机抽样，返回 (seqs, positions_list)。自动避开 'X' 位点。"""
    rng = random.Random(seed)
    pool = []
    if val_path:   pool += _load_jsonl_or_json(val_path, None)
    if train_path: pool += _load_jsonl_or_json(train_path, None)
    if not pool:
        raise RuntimeError("未从数据集读到样本，请检查路径。")
    k_take = min(len(pool), int(n_from_train) + int(n_from_val) if (n_from_train or n_from_val) else 10)
    pool = rng.sample(pool, k_take)

    seqs, pos_list = [], []
    for obj in pool:
        s = obj["seq"]
        cand = [i for i, ch in enumerate(s) if ch != 'X']
        if not cand:
            continue
        k = min(int(n_mut_each), len(cand))
        pos = sorted(rng.sample(cand, k=k))
        seqs.append(s); pos_list.append(pos)
    return seqs, pos_list

def run_iterative_on_dataset(
    mutator: IterativeMutator,
    train_path: Optional[str],
    val_path: Optional[str],
    n_from_train: int = 0,
    n_from_val: int = 10,
    n_mut_each: int = 6,
    k_mutants: int = 1,
    *,
    include_original: bool = False,
    scheme: str = "combo3", alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.7,
    temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
    aa_blacklist: str = "C",
    temperature_schedule: Optional[Tuple[float,float]] = None,
    seed: Optional[int] = 0,
    verbose: bool = True,
    print_summaries: bool = True,
):
    """数据集抽样 + 迭代突变；返回 List[List[(mut_seq, hist)]]."""
    seqs, pos_list = sample_from_dataset(
        train_path, val_path,
        n_from_train=n_from_train, n_from_val=n_from_val,
        n_mut_each=n_mut_each, seed=seed,
    )
    # 在 batch 跑时，mutator 会自动打印分割符
    return run_iterative_for_many(
        mutator, seqs, pos_list, k_mutants=k_mutants,
        include_original=include_original, scheme=scheme, alpha=alpha, beta=beta, gamma=gamma,
        temperature=temperature, top_k=top_k, top_p=top_p,
        aa_blacklist=aa_blacklist, temperature_schedule=temperature_schedule,
        verbose=verbose, print_summaries=print_summaries,
    )

'''
###########抽样示例################
# 1) 初始化
mut = IterativeMutator(
    weights_path="best.pt",
    local_model_dir="/data/linbu/RandomMutation/models/esm2_650m",
)

# 2B) 数据集抽样：多条序列，每条生成 K 个变体，并带有序列间分割符
ds_outs = run_iterative_on_dataset(
    mut,
    train_path="data_v2/train_withX_maskpos.jsonl.gz",
    val_path="data_v2/val_fixed_withX_maskpos.jsonl.gz",
    n_from_val=3, n_from_train=3, n_mut_each=6,
    k_mutants=3, include_original=False,
    scheme="combo", alpha=1.0, beta=1.0, gamma=0.7,
    verbose=True, print_summaries=True
)
'''

#####外部传入数据#######
# 1) 初始化
mut = IterativeMutator(
    weights_path="best.pt",
    local_model_dir="/data/linbu/RandomMutation/models/esm2_650m",
)

# 2) 外部多条序列 + 各自位点（0-based）
seqs = [
    "EVQLVESGGGLVQPGGSLRLSCAASGTT...YYY",
    "QVQLVQSGAEVKKPGASVKVSCKASGGT...YYY",
]
positions_list = [
    [31, 33, 52, 57, 60, 98],  # 对应 seqs[0]
    [30, 32, 50, 56, 62, 95],  # 对应 seqs[1]
]

outs = run_iterative_for_many(
    mut, seqs, positions_list,
    k_mutants=3,                   # 每条生成 3 个变体
    include_original=False,        # 强制突变（排除原氨基酸）
    scheme="combo3", alpha=1.0, beta=1.0, gamma=0.7,  # 位置权重
    temperature=1.0, top_k=8, top_p=None,
    aa_blacklist="C",              # 可空串 "" 表示不禁
    verbose=True, print_summaries=True
)

# 3) 读取结果（每条序列的 K 个变体）
for si, variants in enumerate(outs):
    print(f"\n### Sequence {si} results: K={len(variants)}")
    for vi, (mut_seq, hist) in enumerate(variants):
        changed = sum(h["picked"]["changed"] for h in hist)
        order = [ (h["picked"]["pos"], f'{h["picked"]["prev_char"]}->{h["picked"]["new_char"]}') for h in hist if h["picked"]["changed"] ]
        print(f"  - var#{vi+1}: changed={changed}, order={order}")
        # mut_seq 是最终突变后的序列

