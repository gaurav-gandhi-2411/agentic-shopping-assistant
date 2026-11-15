"""Peak-memory measurement of the index hot-reload swap (Wave 1b V3(b)).

Loads the real unified index (read-only COPIES, never the live data dir) the way the API
lifespan does, then loads a SECOND copy while the first is held (the real swap peak), then drops
the first. RSS via psutil (Windows: working set; peak via peak_wset). The embedding model is a
stub (no torch): MiniLM/CLIP weights are NOT in these numbers.
usage: measure_index_swap_memory.py <data_root> <repo_root> <stub_dir>
<data_root> holds unified/ (the 5 index files) and clip_unified/ (clip.faiss + ids);
<stub_dir> holds a stub sentence_transformers package (no torch needed).
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import psutil

data_root, repo_root, stub_dir = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(stub_dir))
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)
os.environ.pop("INCLUDE_NON_PUBLIC_STORES", None)  # default = public only (production)

import yaml  # noqa: E402

from src.retrieval.snapshot import load_clip_entry, load_index_parts  # noqa: E402

GIB = 1024**3
MIB = 1024**2
proc = psutil.Process()


class FakeModel:
    def encode(self, *a, **k):  # pragma: no cover
        raise RuntimeError("stub")


class Peak:
    """Samples RSS every 10 ms in a thread; peak = max(sampled, OS-reported peak working set)."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak = 0
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, proc.memory_info().rss)
            time.sleep(0.01)

    def __enter__(self) -> "Peak":
        self.t.start()
        return self

    def __exit__(self, *a) -> None:
        self.stop.set()
        self.t.join()
        self.peak = max(self.peak, proc.memory_info().rss)


def rss() -> int:
    return proc.memory_info().rss


def row(label: str, value: int, note: str = "") -> None:
    print(f"{label:<58} {value / MIB:9.0f} MiB  {value / GIB:6.3f} GiB  {note}")


cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
unified = data_root / "unified"
clip_dir = data_root / "clip_unified"
files = [p for p in list(unified.iterdir()) + list(clip_dir.iterdir()) if p.is_file()]
build_bytes = sum(p.stat().st_size for p in files)
print("build files:", {p.name: p.stat().st_size for p in files})
row("build size on disk (incl. CLIP) = tmp bytes on memory-backed FS", build_bytes)

try:
    from src.agents.graph import build_graph  # heavy import, part of the real process

    have_graph = True
except Exception as exc:  # noqa: BLE001
    print("build_graph import failed:", type(exc).__name__, str(exc)[:120])
    have_graph = False

gc.collect()
base = rss()
row("0. baseline after imports (faiss, pandas, langgraph, src.*)", base)


def load_snapshot_parts(tag: str):
    with Peak() as pk:
        retr, df = load_index_parts(unified, cfg, model=FakeModel())
        clip = load_clip_entry(clip_dir)
    row(f"   {tag}: index+catalogue+CLIP loaded, RSS", rss())
    row(f"   {tag}: peak during that load", pk.peak)
    graphs = None
    if have_graph:
        llm = MagicMock()
        with Peak() as pk2:
            graphs = (
                build_graph(retr, df, llm, cfg, streaming_mode=False),
                build_graph(retr, df, llm, cfg, streaming_mode=True),
            )
        row(f"   {tag}: both graphs compiled, RSS", rss())
        row(f"   {tag}: peak during graph compile", pk2.peak)
    return retr, df, clip, graphs


print()
print("1. STEADY STATE (snapshot A, as the lifespan loads it)")
t0 = time.time()
A = load_snapshot_parts("A")
print(
    f"   A load+compile wall time: {time.time() - t0:.1f} s; items={len(A[1])} (public view), "
    f"dense vectors={A[0].dense.index.ntotal}, dim={A[0].dense.index.d}, clip={A[2][0].ntotal}"
)
gc.collect()
steady = rss()
row("1. STEADY RSS (A only, gc'd)", steady)

print()
print("2. SWAP: load B while A is still held (in-flight/serving)")
t0 = time.time()
with Peak() as swap_pk:
    B = load_snapshot_parts("B")
print(f"   B load+compile wall time: {time.time() - t0:.1f} s")
gc.collect()
both = rss()
row("2a. RSS holding A and B (after B loaded, gc'd)", both)
row("2b. PEAK RSS during the whole swap (A held + B loading + compile)", swap_pk.peak)
os_peak = getattr(proc.memory_info(), "peak_wset", 0)
row("2c. OS-reported peak working set of the process", os_peak)
swap_peak = max(swap_pk.peak, os_peak)
row("2d. swap peak + build size as resident tmp files (worst case)", swap_peak + build_bytes)

print()
print("3. AFTER the swap: A dropped")
del A
gc.collect()
after = rss()
row(
    "3. RSS after dropping A + gc.collect()",
    after,
    f"(delta vs steady {((after - steady) / MIB):+.0f} MiB)",
)
del B
gc.collect()
print()
print(
    f"SUMMARY steady={steady / GIB:.3f} GiB  swap_peak={swap_peak / GIB:.3f} GiB  "
    f"swap_peak+tmp={(swap_peak + build_bytes) / GIB:.3f} GiB  after={after / GIB:.3f} GiB"
)
