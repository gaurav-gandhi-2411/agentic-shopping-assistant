"""Container cold start / steady RSS / encode latency for an API image (host-side driver).

    python scripts/bench_container.py --image asa-onnx-bench:onnx --label onnx \
        --processed-dir <data/processed (read-only mount source)> --runs 5 --out onnx.json

Runs the image the way prod does where it matters (port 8080, ``--cpus 1 --memory 4g`` = the
Cloud Run limits in deploy/service.yaml, read-only index volume mounts standing in for the GCS
download, startup warm-up ON) and with NO secrets: ``LLM_PROVIDER=ollama`` so ``/readyz`` makes no
LLM call (no Groq, no network). ``/readyz`` therefore reports ``database: not configured``; the
measured quantity is "time from `docker run` until ``/readyz`` returns 200", which in this setup
means: process started, config + index + retriever loaded, startup warm-up finished, and the CLIP
stack loaded (warm-up runs it before the port opens). It is NOT a Cloud Run cold start.
Containers are named ``asa-bench-<label>-<n>`` and removed by name when done.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(*args: str, check: bool = True) -> str:
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} -> {r.returncode}: {r.stderr.strip()[:500]}")
    return r.stdout


def start(image: str, name: str, port: int, processed: Path) -> float:
    t0 = time.perf_counter()
    sh("docker", "run", "-d", "--name", name, "-p", f"{port}:8080", "--cpus", "1", "--memory", "4g",
       "-e", "LLM_PROVIDER=ollama",
       "-v", f"{processed / 'unified'}:/app/data/processed/unified:ro",
       "-v", f"{processed / 'clip' / 'unified'}:/app/data/processed/clip/unified:ro", image)
    return t0


def wait_ready(name: str, port: int, t0: float, timeout_s: float = 900.0) -> float:
    url = f"http://127.0.0.1:{port}/readyz"
    while time.perf_counter() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            if e.code not in (503,):
                pass
        except OSError:
            pass
        if sh("docker", "inspect", "-f", "{{.State.Running}}", name).strip() != "true":
            raise RuntimeError(f"container {name} exited:\n{sh('docker', 'logs', '--tail', '40', name, check=False)}")
        time.sleep(0.2)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")


def mem_stats(name: str) -> dict[str, float]:
    raw = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name).strip()
    m = re.match(r"([\d.]+)\s*([KMG]i?B)", raw)
    mult = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "kB": 1 / 1024, "MB": 1.0, "GB": 1024.0}
    docker_mb = float(m.group(1)) * mult[m.group(2)] if m else float("nan")
    rss = sh("docker", "exec", name, "sh", "-c", "grep -h VmRSS /proc/[0-9]*/status", check=False)
    rss_mb = sum(int(x) for x in re.findall(r"VmRSS:\s+(\d+)", rss)) / 1024.0
    return {"docker_stats_mem_mb": round(docker_mb, 1), "sum_vmrss_mb": round(rss_mb, 1)}


def summ(xs: list[float]) -> dict[str, float]:
    return {"n": len(xs), "median": round(statistics.median(xs), 2), "min": round(min(xs), 2),
            "max": round(max(xs), 2), "all": [round(x, 2) for x in xs]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--settle-s", type=float, default=15.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    res: dict = {"image": args.image, "label": args.label,
                 "image_size_mb": None, "limits": "--cpus 1 --memory 4g", "runs": []}
    size = sh("docker", "image", "inspect", "-f", "{{.Size}}", args.image).strip()
    res["image_size_mb"] = round(int(size) / 1e6, 1)  # decimal MB, as `docker image ls` reports
    starts, rss_docker, rss_sum = [], [], []
    for i in range(args.runs):
        name = f"asa-bench-{args.label}-{i}"
        sh("docker", "rm", "-f", name, check=False)
        try:
            t0 = start(args.image, name, args.port, args.processed_dir)
            ready = wait_ready(name, args.port, t0)
            time.sleep(args.settle_s)
            mem = mem_stats(name)
            logs = sh("docker", "logs", name, check=False)
            warm = re.findall(r"startup warm-up finished[^\n]*", logs)
            res["runs"].append({"ready_s": round(ready, 2), **mem, "warmup_log": warm[-1] if warm else None})
            starts.append(ready)
            rss_docker.append(mem["docker_stats_mem_mb"])
            rss_sum.append(mem["sum_vmrss_mb"])
        finally:
            sh("docker", "rm", "-f", name, check=False)
    res["cold_start_to_readyz_200_s"] = summ(starts)
    res["steady_mem_docker_stats_mb"] = summ(rss_docker)
    res["steady_sum_vmrss_mb"] = summ(rss_sum)

    name = f"asa-bench-{args.label}-enc"
    sh("docker", "rm", "-f", name, check=False)
    try:
        t0 = start(args.image, name, args.port, args.processed_dir)
        wait_ready(name, args.port, t0)
        time.sleep(args.settle_s)
        qfile = ROOT / "evals" / "fixtures" / "onnx_equivalence" / "queries.json"
        sh("docker", "cp", str(ROOT / "scripts" / "bench_encode_in_container.py"), f"{name}:/tmp/bench_encode.py")
        sh("docker", "cp", str(qfile), f"{name}:/tmp/queries.json")
        out = sh("docker", "exec", "-e", "TQDM_DISABLE=1", name, "python", "/tmp/bench_encode.py", "/tmp/queries.json")
        res["encode_latency"] = json.loads(out.strip().splitlines()[-1])
    finally:
        sh("docker", "rm", "-f", name, check=False)
    args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
