"""Experiment E5: does equation (11) predict the per-thread width XLA chooses on the baseline?

    v2(#) = v2(B) + v2(n) + v2(b) + sum_k v2(L_k),   w = min(4, 2^v2(#))          (11)

The baseline is an array-compiler Metropolis step on the tensor (10), written with jnp.roll as in
section0. Every grid is compiled with XLA's dump enabled; the width is read from the PTX as the
widest f32 global load (`ld.global[.nc].vK.f32`) of the module and of its dominant kernel.

Run as its own process (the dump flag must be set before JAX starts):
    python -m Spin_Colour_Kernel.width_law --out results/ [--quick]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path


def v2(x: int) -> int:
    k = 0
    while x % 2 == 0:
        x //= 2
        k += 1
    return k


def predicted(B, n, b, box) -> int:
    e = v2(B) + v2(n) + v2(b) + sum(v2(L) for L in box)
    return min(4, 2 ** e)


LD = re.compile(r"\b(ld|st)\.global(?:\.[a-z]+)*?(?:\.v(\d))?\.(f32|b32)\b")


def ptx_widths(text: str) -> dict:
    """Per-kernel histogram of 32-bit (f32, or b32 as Triton emits it) global load and store widths."""
    kernels = re.split(r"\.entry\s+", text)[1:]
    out = []
    for k in kernels:
        name = k.split("(")[0].strip()
        hist = {}
        for op, w, ty in LD.findall(k):
            key = f"{op}.v{w or 1}"
            hist[key] = hist.get(key, 0) + 1
        out.append({"kernel": name, "instructions": k.count(";"), "hist": hist})
    return out


LL_LD = re.compile(r"\b(load|store)\s+(?:<(\d+)\s+x\s+(?:float|i32)>|(?:float|i32)\b)(?![^\n]*addrspace\((?:3|5)\))[^\n]*?\bptr\b")


def ll_widths(text: str) -> list[dict]:
    """Same histogram from XLA's LLVM IR dump: global (addrspace 1) float loads/stores by vector width."""
    out = []
    for k in re.split(r"\ndefine\s", text)[1:]:
        name = k.split("(")[0].split("@")[-1].strip()
        hist = {}
        for op, w in LL_LD.findall(k):
            key = f"{'ld' if op == 'load' else 'st'}.v{w or 1}"
            hist[key] = hist.get(key, 0) + 1
        out.append({"kernel": name, "instructions": k.count("\n"), "hist": hist})
    return out


SASS_LDG = re.compile(r"\bLDG(?:\.[A-Z0-9_]+)*")


def _payloads(data: bytes, depth: int = 0) -> list[bytes]:
    """The entry itself plus every zstd or zlib stream found at any offset, recursively (depth 2)."""
    import zlib
    out = [data]
    if depth >= 2:
        return out
    try:
        import zstandard
        i = data.find(b"\x28\xb5\x2f\xfd")
        while i >= 0:
            try:
                d = zstandard.ZstdDecompressor().decompressobj().decompress(data[i:])
                if d:
                    out += _payloads(d, depth + 1)
            except Exception:
                pass
            i = data.find(b"\x28\xb5\x2f\xfd", i + 4)
    except ImportError:
        pass
    for i in range(len(data) - 1):
        if data[i] == 0x78 and data[i + 1] in (0x01, 0x5E, 0x9C, 0xDA):
            try:
                d = zlib.decompressobj().decompress(data[i:])
                if len(d) > 64:
                    out += _payloads(d, depth + 1)
            except Exception:
                pass
    return out


FATBIN_MAGIC = b"\x50\xed\x55\xba"


def _fatbins(data: bytes) -> list[bytes]:
    """CUDA fatbin containers (magic 0xBA55ED50; 16-byte header, 8-byte payload size at offset 8)."""
    import struct
    out = []
    for d in _payloads(data):
        i = d.find(FATBIN_MAGIC)
        while i >= 0:
            try:
                hsize = struct.unpack_from("<H", d, i + 6)[0]
                fsize = struct.unpack_from("<Q", d, i + 8)[0]
                if 0 < hsize + fsize <= len(d) - i:
                    out.append(d[i:i + hsize + fsize])
            except Exception:
                pass
            i = d.find(FATBIN_MAGIC, i + 4)
    return out


def _elf_blobs(data: bytes) -> list[bytes]:
    """ELF images inside an entry or any stream it compresses; size from the section-header table."""
    import struct
    out = []
    for d in _payloads(data):
        i = d.find(b"\x7fELF")
        while i >= 0:
            try:
                shoff = struct.unpack_from("<Q", d, i + 0x28)[0]
                shentsize, shnum = struct.unpack_from("<HH", d, i + 0x3A)
                size = shoff + shentsize * shnum
                if 64 < size <= len(d) - i:
                    out.append(d[i:i + size])
            except Exception:
                pass
            i = d.find(b"\x7fELF", i + 4)
    return out


def sass_widths(cache: Path, verbose: bool = False):
    """Per-kernel LDG widths from the SASS of every cubin in the persistent compilation cache."""
    import subprocess
    tool = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    ks, files = [], []
    for f in sorted(glob.glob(str(cache / "**" / "*"), recursive=True)):
        if not os.path.isfile(f):
            continue
        raw = Path(f).read_bytes()
        blobs = _elf_blobs(raw) + _fatbins(raw)
        # PTX kept as text inside the entry (some XLA builds serialize PTX, not a cubin)
        for d in _payloads(raw):
            t = d.decode("latin-1")
            if ".entry" in t and "ld.global" in t:
                k = [x for x in ptx_widths(t) if x["hist"]]
                ks.extend(k)
                files.append(f + ":ptx")
        if verbose:
            print(f"E5 cache entry {os.path.basename(f)[:60]} {len(raw)}B payloads={len(_payloads(raw))} "
                  f"elf+fatbin={len(blobs)} head={raw[:16].hex()}", flush=True)
            for d in _payloads(raw)[1:]:
                print("E5   payload", len(d), "fatbin" if FATBIN_MAGIC in d else "", "sm_75" if b"sm_75" in d else "",
                      "ptx" if b".version" in d else "", d[:24].hex(), flush=True)
        for j, blob in enumerate(blobs):
            cub = Path(f + f".{j}.cubin")
            cub.write_bytes(blob)
            r = subprocess.run([tool, "-sass", str(cub)], capture_output=True, text=True)
            if verbose and j == 0:
                print("E5 cuobjdump rc", r.returncode, r.stderr[:300], flush=True)
            for part in re.split(r"\n\s*Function\s*:\s*", r.stdout)[1:]:
                name = part.split("\n")[0].strip()
                hist = {}
                for ins in SASS_LDG.findall(part):
                    w = 4 if ".128" in ins else (2 if ".64" in ins else 1)
                    hist[f"ld.v{w}"] = hist.get(f"ld.v{w}", 0) + 1
                ks.append({"kernel": name, "instructions": part.count(";"), "hist": hist})
            files.append(str(cub))
    return ks, files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--method", choices=("dump", "sass"), default="dump",
                    help="dump: PTX/LLVM-IR from --xla_dump_to; sass: cubins from the persistent compilation cache")
    ap.add_argument("--tag", default="E5")
    a = ap.parse_args(argv)
    dump = Path(tempfile.mkdtemp(prefix="xla_dump_"))
    cache = Path(tempfile.mkdtemp(prefix="jax_cache_"))
    if a.method == "dump":
        extra = os.environ.get("SCK_XLA_EXTRA", "--xla_gpu_dump_llvmir=true")
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={dump} {extra}"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import numpy as np

    if a.method == "sass":
        jax.config.update("jax_compilation_cache_dir", str(cache))
        for opt, val in (("jax_persistent_cache_min_compile_time_secs", 0.0),
                         ("jax_persistent_cache_min_entry_size_bytes", 0)):
            try:
                jax.config.update(opt, val)
            except Exception:
                pass

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Spin_Colour_Kernel import colouring as co
    from Spin_Colour_Kernel import crystals as cr

    crystal_for_b = {1: "sc", 2: "bcc", 4: "fcc"}

    def make_step(n, b, box, gid):
        c = cr.make(crystal_for_b[b])
        col = co.search(c)
        L1, L2, L3 = box
        cm = np.zeros((b, L3, L2, L1), np.int32)
        for al in range(b):
            for z, y, x in itertools.product(range(L3), range(L2), range(L1)):
                cm[al, z, y, x] = col.colour((x, y, z), al)
        cmj = jnp.asarray(cm)
        nc = col.n_colours

        def step(S, key, beta):
            for colour in range(nc):
                vs = []
                for al in range(b):
                    v = jnp.zeros_like(S[:, :, 0])
                    for bd in c.bonds:
                        if bd.alpha != al:
                            continue
                        src = S[:, :, bd.alpha2]
                        for k, ax in zip(range(3), (4, 3, 2)):
                            if bd.delta[k]:
                                src = jnp.roll(src, -bd.delta[k], axis=ax)
                        v = v + src
                    vs.append(v)
                V = jnp.stack(vs, axis=2)
                key, k1, k2 = jax.random.split(key, 3)
                if n == 1:
                    T = -S
                else:
                    g = jax.random.normal(k1, S.shape, S.dtype)
                    T = g / jnp.linalg.norm(g, axis=1, keepdims=True)
                dE = -jnp.sum((T - S) * V, axis=1)
                u = jax.random.uniform(k2, dE.shape, dE.dtype)
                acc = (u < jnp.exp(-beta * dE)) & (cmj[None] == colour)
                S = jnp.where(acc[:, None], T, S)
            return S, key

        step.__name__ = f"wl{gid:04d}"
        return jax.jit(step)

    if a.quick:
        grids = [(B, 3, 1, (5, 5, 5)) for B in (1, 2, 3, 4)] + [(1, n, 1, (5, 5, 5)) for n in (1, 2)]
    else:
        grids = [(B, n, b, (L1, 5, 5)) for B in (1, 2, 3, 4, 6, 8) for n in (1, 2, 3) for b in (1, 2, 4)
                 for L1 in (5, 6, 8)]
    rows = []
    t0 = time.time()
    for gid, (B, n, b, box) in enumerate(grids):
        L1, L2, L3 = box
        for d in (dump, cache):                       # every file now belongs to this grid
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        f = make_step(n, b, box, gid)
        S = jnp.ones((B, n, b, L3, L2, L1), jnp.float32) / np.sqrt(n)
        key = jax.random.PRNGKey(gid)
        S2, key = f(S, key, 0.5)
        S2.block_until_ready()
        # time per update over 20 calls
        t = time.perf_counter()
        for _ in range(20):
            S2, key = f(S2, key, 0.5)
        S2.block_until_ready()
        dt = (time.perf_counter() - t) / 20
        every = sorted(f for f in glob.glob(str(dump / "**" / "*"), recursive=True) if os.path.isfile(f))
        if gid == 0 and every:
            lls = [f for f in every if f.endswith(".ll")] or every
            big = max(lls, key=os.path.getsize)
            lines = [l.strip()[:160] for l in Path(big).read_text(errors="ignore").splitlines()
                     if re.search(r"\b(load|store)\b|ld\.global|vector\.load|llvm\.load", l)][:12]
            print("E5 largest dump file", os.path.basename(big)[-80:], os.path.getsize(big), "B; global accesses:",
                  lines, flush=True)
        files, ks, source = [], [], None
        if a.method == "sass":
            ks, files = sass_widths(cache, gid == 0)
            source = "sass" if ks else None
            every = []
        for f in every:
            try:
                t = Path(f).read_text(errors="ignore")
            except Exception:
                continue
            kp = [k for k in ptx_widths(t) if any(h.startswith("ld") for h in k["hist"])]
            kl = [] if kp else [k for k in ll_widths(t) if any(h.startswith("ld") for h in k["hist"])]
            if gid == 0 and (t.count("ld.global") or LL_LD.search(t) or f.endswith((".ptx", ".ll"))):
                print(f"E5 file {os.path.basename(f)[-70:]:70s} {len(t):9d}B ld.global={t.count('ld.global')} "
                      f"llvm-global-loads={len(LL_LD.findall(t))}", flush=True)
            if kp or kl:
                files.append(f)
                ks.extend(kp or kl)
                source = source or ("ptx" if kp else "llvm-ir")
        loads = [(k, int(key.split(".v")[1])) for k in ks for key in k["hist"] if key.startswith("ld")]
        module_w = max((w for _, w in loads), default=None)
        dom = max((k for k in ks if any(h.startswith("ld") for h in k["hist"])), key=lambda k: k["instructions"],
                  default=None)
        dom_w = max((int(h.split(".v")[1]) for h in dom["hist"] if h.startswith("ld")), default=None) if dom else None
        pred = predicted(B, n, b, box)
        rows.append({"B": B, "n": n, "b": b, "box": list(box), "elements": B * n * b * L1 * L2 * L3,
                     "v2": v2(B) + v2(n) + v2(b) + sum(v2(L) for L in box), "predicted": pred,
                     "measured_module": module_w, "measured_dominant": dom_w,
                     "agree_module": module_w == pred, "agree_dominant": dom_w == pred,
                     "seconds_per_step": dt, "ns_per_update": dt * 1e9 / (B * b * L1 * L2 * L3),
                     "source": source, "files": len(files), "kernels": ks})
        print(f"E5 grid {gid:3d} B={B} n={n} b={b} box={box} pred={pred} module={module_w} dom={dom_w} "
              f"{time.time() - t0:.0f}s", flush=True)
        for p in glob.glob(str(dump / f"*wl{gid:04d}*")):
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)

    def factor(rows, key):
        base = {"B": lambda r: r["n"] in (1, 3) and r["b"] == 1 and r["box"][0] == 5,
                "n": lambda r: r["B"] in (1, 3) and r["b"] == 1 and r["box"][0] == 5,
                "b": lambda r: r["B"] in (1, 3) and r["n"] in (1, 3) and r["box"][0] == 5,
                "box": lambda r: r["B"] in (1, 3) and r["n"] in (1, 3) and r["b"] == 1}[key]
        sel = [r for r in rows if base(r)]
        return {"grids": len(sel), "agree_module": sum(r["agree_module"] for r in sel),
                "agree_dominant": sum(r["agree_dominant"] for r in sel)}

    summary = {k: factor(rows, k) for k in ("B", "n", "b", "box")}
    summary["all"] = {"grids": len(rows), "agree_module": sum(r["agree_module"] for r in rows),
                      "agree_dominant": sum(r["agree_dominant"] for r in rows),
                      "unreadable": sum(r["measured_module"] is None for r in rows)}
    a.out.mkdir(parents=True, exist_ok=True)
    import platform
    payload = {"experiment": a.tag, "method": a.method, "rows": rows, "summary": summary, "jax": jax.__version__,
               "environment": {"python": platform.python_version(), "jax": jax.__version__,
                               "time": time.strftime("%Y-%m-%dT%H:%M:%S")},
               "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()]}
    try:
        import jaxlib
        payload["jaxlib"] = jaxlib.__version__
    except Exception:
        pass
    (a.out / f"{a.tag}.json").write_text(json.dumps(payload, indent=2, default=float))
    shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(dump, ignore_errors=True)
    print(a.tag, "summary", json.dumps(summary))


if __name__ == "__main__":
    main()
