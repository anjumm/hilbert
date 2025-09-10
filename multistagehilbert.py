# multi_stage_hilbert_router.py
# Requires: pip install hilbertcurve pandas numpy

import json
import math
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
from hilbertcurve.hilbertcurve import HilbertCurve

# ===============================
# ======= CONFIGURATION =========
# ===============================

DATA_FILE = r"C:\Users\malik\Desktop\pythonscripts\rttsfor80nodes.txt"
NET_P_BITS = 14                               # Hilbert bits for network geometry (primary)
RES_P_BITS = 7                                # Hilbert bits for resources (secondary)
K_LANDMARKS = 6                               # used only if coordinates are missing; pick K landmarks
RTT_PCTL_FOR_DELTA = 95                       # map RTT threshold -> Δ via percentile of primary distances
VERIFY_LATENCY_VIA_RTT = True                 # after primary window, optionally verify with measured RTTs

# Resource tiers (deterministic assignment based on node name hash)
# Buckets define % of nodes (from hash 0..99) mapped to each tier.
RESOURCE_TIERS = {
    "T3": {"cpu": 32, "ram": 64, "storage": 2000, "bucket": range(0, 10)},    # ~10%
    "T2": {"cpu": 24, "ram": 48, "storage": 1500, "bucket": range(10, 30)},    # ~20%
    "T1": {"cpu": 16, "ram": 32, "storage": 1000, "bucket": range(30, 60)},    # ~30%
    "T0": {"cpu": 8,  "ram": 16, "storage": 500,  "bucket": range(60, 100)},   # ~40%
}

# ===============================
# =========== TYPES =============
# ===============================

@dataclass
class NodeRec:
    name: str
    coord: Optional[List[float]]  # Serf/Vivaldi coords (d dims) if available
    rtts: Dict[str, float]        # measured RTTs to peers (ms)
    cpu: int
    ram: int
    storage: int
    h_net: int                    # primary Hilbert key
    h_res: int                    # secondary Hilbert key (resources)

# ===============================
# ======== HELPERS ==============
# ===============================

def extract_serf_number(name: str) -> int:
    m = re.search(r"serf(\d+)", name)
    return int(m.group(1)) if m else 10**9

def deter_bucket_0_99(name: str) -> int:
    # stable 0..99 from name
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100

def assign_resources(name: str) -> Tuple[int, int, int, str]:
    b = deter_bucket_0_99(name)
    for tier_name, spec in RESOURCE_TIERS.items():
        if b in spec["bucket"]:
            return spec["cpu"], spec["ram"], spec["storage"], tier_name
    # fallback
    spec = RESOURCE_TIERS["T0"]
    return spec["cpu"], spec["ram"], spec["storage"], "T0"

def minmax_norm_to_bits(values: np.ndarray, p_bits: int) -> np.ndarray:
    """Normalize each dimension independently to [0, 2^p_bits - 1]."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (values - lo) / span
    return np.round(scaled * ((2 ** p_bits) - 1)).astype(int), lo, hi

def norm_with_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, p_bits: int) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (x - lo) / span
    return np.clip(np.round(scaled * ((2 ** p_bits) - 1)), 0, (2 ** p_bits) - 1).astype(int)

def build_landmark_vectors(nodes: List[dict], k: int) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    If no coordinates exist, we fallback to K-landmark RTT vectors per node.
    Landmarks chosen deterministically by smallest serf number across the set.
    """
    names = [n["name"] for n in nodes]
    names_sorted = sorted(names, key=extract_serf_number)
    landmarks = names_sorted[:k]
    lm_vectors: Dict[str, np.ndarray] = {}
    for n in nodes:
        vec = []
        rtts = n.get("rtts", {})
        for lm in landmarks:
            vec.append(float(rtts.get(lm, np.nan)))
        lm_vectors[n["name"]] = np.array(vec, dtype=float)
    # fill NaNs with column medians (robust)
    mat = np.vstack([lm_vectors[n] for n in names])
    col_med = np.nanmedian(mat, axis=0)
    mat = np.where(np.isnan(mat), col_med, mat)
    for i, nm in enumerate(names):
        lm_vectors[nm] = mat[i]
    return lm_vectors, landmarks

def pick_coords_or_landmarks(nodes: List[dict]) -> Tuple[Dict[str, np.ndarray], str, List[str]]:
    """
    Return a mapping name -> geometry vector (either coords or landmark RTTs)
    and a label describing the geometry type.
    """
    coords_exist = all("coordinate" in n and n["coordinate"] and "Vec" in n["coordinate"] for n in nodes)
    if coords_exist:
        geom = {n["name"]: np.array(n["coordinate"]["Vec"], dtype=float) for n in nodes}
        return geom, "coords", []
    lm_vecs, landmarks = build_landmark_vectors(nodes, K_LANDMARKS)
    return lm_vecs, "landmarks", landmarks

def percentile_safe(arr: Iterable[int], p: int) -> Optional[int]:
    arr = list(arr)
    if not arr:
        return None
    return int(np.percentile(arr, p))

# ===============================
# ======= PIPELINE CORE =========
# ===============================

class MultiStageHilbert:
    def __init__(self,
                 nodes: List[dict],
                 net_p_bits: int = NET_P_BITS,
                 res_p_bits: int = RES_P_BITS):
        self.nodes_raw = nodes
        self.net_p_bits = net_p_bits
        self.res_p_bits = res_p_bits

        # --- geometry source
        self.geom_map, self.geom_kind, self.landmarks = pick_coords_or_landmarks(nodes)

        # --- collect geometry matrix
        self.names = [n["name"] for n in nodes]
        self.geom = np.vstack([self.geom_map[nm] for nm in self.names])  # shape N x D
        self.N, self.D = self.geom.shape

        # --- normalize geometry to hilbert grid & build H_net
        self.geom_norm, self.geom_lo, self.geom_hi = minmax_norm_to_bits(self.geom, self.net_p_bits)
        self.HNET = HilbertCurve(p=self.net_p_bits, n=self.D)
        h_net_vals = [self.HNET.distance_from_point(self.geom_norm[i].tolist()) for i in range(self.N)]

        # --- assign resources & build H_res
        cpu_list, ram_list, sto_list, tier_list = [], [], [], []
        for nm in self.names:
            cpu, ram, sto, tier = assign_resources(nm)
            cpu_list.append(cpu); ram_list.append(ram); sto_list.append(sto); tier_list.append(tier)
        self.cpu = np.array(cpu_list); self.ram = np.array(ram_list); self.sto = np.array(sto_list)
        self.tier = np.array(tier_list)

        # normalize resources for H_res
        res_mat = np.vstack([self.cpu, self.ram, self.sto]).T
        self.res_norm, self.res_lo, self.res_hi = minmax_norm_to_bits(res_mat, self.res_p_bits)
        self.HRES = HilbertCurve(p=self.res_p_bits, n=3)
        h_res_vals = [self.HRES.distance_from_point(self.res_norm[i].tolist()) for i in range(self.N)]

        # --- measured RTTs dictionary
        self.rtts: Dict[str, Dict[str, float]] = {n["name"]: {k: float(v) for k, v in n.get("rtts", {}).items()} for n in nodes}

        # --- dataframe view
        self.df = pd.DataFrame({
            "name": self.names,
            "h_net": h_net_vals,
            "h_res": h_res_vals,
            "cpu": self.cpu,
            "ram": self.ram,
            "storage": self.sto,
            "tier": self.tier,
        })

        # --- primary ordering
        self.df.sort_values("h_net", inplace=True, kind="mergesort")  # stable
        self.df.reset_index(drop=True, inplace=True)
        self.name_to_idx = {name: idx for idx, name in self.df["name"].items()}

        # --- tier views: indices (in primary order) per tier
        self.tier_views: Dict[str, np.ndarray] = {}
        for t in np.unique(self.df["tier"]):
            self.tier_views[t] = self.df.index[self.df["tier"] == t].to_numpy()

        # --- precompute avg RTT per node (optional)
        self.avg_rtt = {}
        for nm in self.names:
            vals = [v for v in self.rtts.get(nm, {}).values() if v is not None]
            self.avg_rtt[nm] = float(np.mean(vals)) if vals else float("nan")

    # ---------- Utilities ----------

    def _get_hnet(self, name: str) -> int:
        return int(self.df.at[self.name_to_idx[name], "h_net"])

    def _map_rtt_to_delta(self, query: str, rtt_ms: float, percentile: int = RTT_PCTL_FOR_DELTA) -> Optional[int]:
        """
        Build a per-query mapping from RTT<=threshold to |ΔH_net| percentile.
        """
        q_h = self._get_hnet(query)
        lat = self.rtts.get(query, {})
        dists = []
        for peer, rtt in lat.items():
            if rtt is None or math.isnan(rtt):
                continue
            if rtt <= rtt_ms and peer in self.name_to_idx:
                d = abs(int(self.df.at[self.name_to_idx[peer], "h_net"]) - q_h)
                dists.append(d)
        return percentile_safe(dists, percentile)

    def _resource_ok(self, idx: int, min_cpu: int, min_ram: int, min_storage: int) -> bool:
        r = self.df.loc[idx]
        return (r.cpu >= min_cpu) and (r.ram >= min_ram) and (r.storage >= min_storage)

    def _optional_verify_latency(self, query: str, cand_names: List[str], max_rtt: Optional[float]) -> List[str]:
        if not VERIFY_LATENCY_VIA_RTT or max_rtt is None:
            return cand_names
        lat = self.rtts.get(query, {})
        return [nm for nm in cand_names if lat.get(nm) is not None and lat[nm] <= max_rtt]

    # ---------- Public API ----------

    def range_query(self,
                    query_name: str,
                    rtt_threshold_ms: float,
                    min_cpu: int = 0,
                    min_ram: int = 0,
                    min_storage: int = 0,
                    prefer_tier: Optional[str] = None,
                    k: Optional[int] = None,
                    rank_mode: str = "hres"  # 'hres' | 'score' | 'none'
                   ) -> pd.DataFrame:
        """
        Two-stage query:
          1) primary window in H_net mapped from RTT threshold (Δ via per-query calibration)
          2) resource constraints (and optional ranking)
        """
        if query_name not in self.name_to_idx:
            raise ValueError(f"Unknown node: {query_name}")

        delta = self._map_rtt_to_delta(query_name, rtt_threshold_ms)
        if delta is None:
            # Fallback: broaden until we find anything (progressive doubling)
            delta = 1
            q_h = self._get_hnet(query_name)
            for _ in range(18):  # up to ~2^18 window if really needed
                window = (q_h - delta, q_h + delta)
                cand = self._candidates_in_window(window, prefer_tier)
                cand = [c for c in cand if self._resource_ok(self.name_to_idx[c], min_cpu, min_ram, min_storage)]
                cand = self._optional_verify_latency(query_name, cand, rtt_threshold_ms)
                if cand:
                    break
                delta *= 2

        q_h = self._get_hnet(query_name)
        window = (q_h - delta, q_h + delta)

        cand = self._candidates_in_window(window, prefer_tier)
        # filter out self
        cand = [c for c in cand if c != query_name]

        # resource filter
        cand = [c for c in cand if self._resource_ok(self.name_to_idx[c], min_cpu, min_ram, min_storage)]

        # optional RTT verify
        cand = self._optional_verify_latency(query_name, cand, rtt_threshold_ms)

        # ranking
        if rank_mode == "hres":
            cand.sort(key=lambda nm: int(self.df.at[self.name_to_idx[nm], "h_res"]))
        elif rank_mode == "score":
            # higher resources first (cpu,ram,storage), break ties by h_res
            cand.sort(key=lambda nm: (-int(self.df.at[self.name_to_idx[nm], "cpu"]),
                                      -int(self.df.at[self.name_to_idx[nm], "ram"]),
                                      -int(self.df.at[self.name_to_idx[nm], "storage"]),
                                      int(self.df.at[self.name_to_idx[nm], "h_res"])))
        else:
            # default order by primary closeness
            cand.sort(key=lambda nm: abs(int(self.df.at[self.name_to_idx[nm], "h_net"]) - q_h))

        if k is not None:
            cand = cand[:k]

        out = self.df.set_index("name").loc[cand, ["cpu", "ram", "storage", "tier", "h_net", "h_res"]].copy()
        # add distances for transparency
        out["dH"] = [abs(int(out.at[nm, "h_net"]) - q_h) for nm in out.index]
        out["rtt_to_query"] = [self.rtts.get(query_name, {}).get(nm, np.nan) for nm in out.index]
        return out.sort_values(["dH", "h_res"])

    def _candidates_in_window(self, window: Tuple[int, int], prefer_tier: Optional[str]) -> List[str]:
        lo, hi = window
        # because h_net is sorted, use boolean mask; wrap-around not needed (Hilbert domain is linearized)
        if prefer_tier and prefer_tier in self.tier_views:
            idxs = self.tier_views[prefer_tier]
            sub = self.df.loc[idxs]
        else:
            sub = self.df
        mask = (sub["h_net"] >= lo) & (sub["h_net"] <= hi)
        return sub.loc[mask, "name"].tolist()

    # ---------- Evaluation ----------

    def test_accuracy(self,
                      query_name: str,
                      rtt_threshold_ms: float,
                      **kwargs) -> Dict[str, float]:
        """
        Compare result set against ground-truth RTT<=threshold for the query node.
        """
        gt_rtts = self.rtts.get(query_name, {})
        true_set = {peer for peer, r in gt_rtts.items() if (r is not None and r <= rtt_threshold_ms)}

        res = self.range_query(query_name, rtt_threshold_ms, **kwargs)
        pred_set = set(res.index)

        inter = pred_set & true_set
        union = pred_set | true_set

        precision = (len(inter) / len(pred_set)) if pred_set else 0.0
        recall = (len(inter) / len(true_set)) if true_set else 0.0
        jaccard = (len(inter) / len(union)) if union else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "jaccard": jaccard,
            "pred_size": float(len(pred_set)),
            "true_size": float(len(true_set))
        }

# ===============================
# =========== LOADER ============
# ===============================

def load_nodes(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)

# ===============================
# ============ MAIN =============
# ===============================

if __name__ == "__main__":
    # 1) Load data
    nodes = load_nodes(DATA_FILE)

    # 2) Build router
    router = MultiStageHilbert(nodes)

    # 3) Show a quick summary
    print(f"Geometry kind: {router.geom_kind} (dims={router.D})"
          + (f", landmarks={router.landmarks}" if router.geom_kind == "landmarks" else ""))
    print(router.df.head(10).to_string(index=False))

    # 4) Example queries
    q = "clab-century-serf1"  # change as needed to any present node

    # Low-latency neighborhood (e.g., 20ms), modest resources
    result1 = router.range_query(
        query_name=q,
        rtt_threshold_ms=20,
        min_cpu=12,
        min_ram=24,
        min_storage=600,
        prefer_tier=None,   # or "T3"/"T2"/"T1"/"T0" to force tier
        k=10,
        rank_mode="hres"    # or "score" / "none"
    )
    print("\n--- Query #1 (<=20ms, modest resources, top-10) ---")
    print(result1.to_string())

    # Stricter latency, higher resources, resource-score ranking
    result2 = router.range_query(
        query_name=q,
        rtt_threshold_ms=8,
        min_cpu=24,
        min_ram=48,
        min_storage=1200,
        prefer_tier="T2",   # try a high tier first; if empty, just switch prefer_tier=None
        k=8,
        rank_mode="score"
    )
    print("\n--- Query #2 (<=8ms, higher resources, tier=T2, top-8, score-ranked) ---")
    print(result2.to_string())

    # Accuracy probe across a few RTT cutoffs
    for thr in [5, 10, 20, 40, 80, 120]:
        metrics = router.test_accuracy(q, thr, min_cpu=0, min_ram=0, min_storage=0, k=None, rank_mode="none")
        print(f"\nAccuracy @ RTT<={thr}ms: "
              f"prec={metrics['precision']:.2f} rec={metrics['recall']:.2f} jaccard={metrics['jaccard']:.2f} "
              f"(pred={int(metrics['pred_size'])}, true={int(metrics['true_size'])})")
