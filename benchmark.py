import copy
import json
import os
import time
import statistics
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    from IPython.display import display
except ImportError:
    display = print

NUM_DEVICES = 2
OBJECTIVE = "latency"
BATCH = 1
IMG_SIZE = 224
WARMUP = 10
INNER = 20
REPEATS = 5
MEASURE_ITERS = 20
MIN_GAIN = 0.05
RUN_SWEEP = True
SPEED_MULTS = [1, 2, 4, 8, 16, 32]
LINK_MULTS = [0.1, 0.3, 1, 3, 10, 30]

NET_LATENCY_MS = 0.5
NET_BANDWIDTH_MBPS = 100.0

DEVICE_POOL = [
    dict(name="cpu",      torch="cpu",  threads=None),
    dict(name="gpu",      torch="cuda", threads=None),
    dict(name="cpu-1t",   torch="cpu",  threads=1),
    dict(name="cpu-1t-b", torch="cpu",  threads=1),
]

SEED = 0
SPECS = []
ACT_BYTES = []


def conv_bn_relu(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def build_units():
    chans = [3, 32, 64, 128, 256, 512]
    units = []
    for cin, cout in zip(chans[:-1], chans[1:]):
        units.append((f"conv {cin}->{cout} s2", conv_bn_relu(cin, cout, 2)))
        units.append((f"conv {cout}->{cout} s1", conv_bn_relu(cout, cout, 1)))
    units.append(("avgpool+flatten", nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())))
    units.append(("fc 512->2", nn.Linear(512, 2)))
    units = [(f"L{i + 1} {n}", m.eval()) for i, (n, m) in enumerate(units)]
    return units


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def time_unit(unit, x, dev):
    x = x.to(dev)
    for _ in range(WARMUP):
        unit(x)
    sync(dev)
    blocks = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(INNER):
            unit(x)
        sync(dev)
        blocks.append((time.perf_counter() - t0) / INNER)
    return statistics.median(blocks) * 1000


def select_devices():
    pool = []
    for spec in DEVICE_POOL:
        if spec["torch"] == "cuda" and not torch.cuda.is_available():
            print(f"skipping '{spec['name']}': no GPU in this runtime")
            continue
        if "profile_file" in spec and not os.path.exists(spec["profile_file"]):
            print(f"skipping '{spec['name']}': {spec['profile_file']} not found")
            continue
        pool.append(dict(spec))
    if len(pool) < NUM_DEVICES:
        print(f"warning: only {len(pool)} devices available, wanted {NUM_DEVICES}")
    return pool[:NUM_DEVICES]


def profile_all(specs, units, acts):
    L = len(units)
    T, dev_units = [], {}
    for i, spec in enumerate(specs):
        if "profile_file" in spec:
            with open(spec["profile_file"]) as f:
                data = json.load(f)
            assert len(data["times_ms"]) == L, "profile has a different number of layers"
            if data.get("batch") != BATCH:
                print(f"warning: {spec['profile_file']} was measured at batch {data.get('batch')}, config has {BATCH}")
            times = data["times_ms"]
            spec["executable"] = False
        else:
            dev = torch.device(spec["torch"])
            old = torch.get_num_threads()
            if dev.type == "cpu" and spec.get("threads"):
                torch.set_num_threads(spec["threads"])
            dev_units[i] = [copy.deepcopy(u).to(dev).eval() for _, u in units]
            print(f"profiling {spec['name']} ...")
            times = [time_unit(dev_units[i][l], acts[l], dev) for l in range(L)]
            torch.set_num_threads(old)
            spec["executable"] = True
            with open(f"profile_{spec['name']}.json", "w") as f:
                json.dump({"name": spec["name"], "batch": BATCH, "img_size": IMG_SIZE,
                           "layers": [n for n, _ in units], "times_ms": times}, f, indent=1)
        T.append(times)
    return T, dev_units


_copy_cache = {}


def copy_ms(nbytes, src, dst):
    key = (src, dst, nbytes)
    if key not in _copy_cache:
        s, d = torch.device(src), torch.device(dst)
        x = torch.empty(max(1, nbytes // 4), device=s)
        ts = []
        for _ in range(12):
            sync(s)
            t0 = time.perf_counter()
            x.to(d)
            sync(d)
            ts.append(time.perf_counter() - t0)
        _copy_cache[key] = statistics.median(ts[2:]) * 1000
    return _copy_cache[key]


def link_cost(a, b, nbytes):
    if a == b:
        return 0.0, "same"
    ta, tb = SPECS[a]["torch"], SPECS[b]["torch"]
    if SPECS[a]["executable"] and SPECS[b]["executable"] and {ta, tb} == {"cpu", "cuda"}:
        return copy_ms(nbytes, ta, tb), "measured"
    return NET_LATENCY_MS + nbytes / (NET_BANDWIDTH_MBPS * 1e6) * 1000, "simulated"


def evaluate(assign, T, X, start=0):
    L = len(assign)
    stages, prev = [], start
    for l, d in enumerate(assign):
        if not stages or stages[-1]["dev"] != d:
            tx = X[prev][d][l]
            stages.append(dict(dev=d, start=l, end=l, compute=0.0, tx_in=tx,
                               tx_bytes=ACT_BYTES[l] if tx > 0 else 0))
        stages[-1]["end"] = l
        stages[-1]["compute"] += T[d][l]
        prev = d
    ret = X[prev][start][L]
    for s in stages:
        s["total"] = s["compute"] + s["tx_in"]
    return dict(stages=stages, ret=ret,
                latency=sum(s["total"] for s in stages) + ret,
                bottleneck=max(s["total"] for s in stages))


def plan_latency(T, X, N, L, start):
    INF = float("inf")
    best = [[INF] * N for _ in range(L)]
    back = [[None] * N for _ in range(L)]
    for d in range(N):
        best[0][d] = X[start][d][0] + T[d][0]
    for l in range(1, L):
        for d in range(N):
            for p in range(N):
                c = best[l - 1][p] + X[p][d][l] + T[d][l]
                if c < best[l][d]:
                    best[l][d], back[l][d] = c, p
    d = min(range(N), key=lambda i: best[L - 1][i] + X[i][start][L])
    assign = [0] * L
    for l in range(L - 1, -1, -1):
        assign[l] = d
        if l > 0:
            d = back[l][d]
    return assign


def plan_throughput(T, X, N, L, start):
    P = [[0.0] * (L + 1) for _ in range(N)]
    for d in range(N):
        for k in range(L):
            P[d][k + 1] = P[d][k] + T[d][k]
    dp = {}
    for d in range(N):
        for k in range(1, L + 1):
            dp[(1 << d, d, k)] = (X[start][d][0] + P[d][k], None)
    for j in range(1, L):
        for mask in range(1, 1 << N):
            for last in range(N):
                key = (mask, last, j)
                if key not in dp:
                    continue
                val = dp[key][0]
                for d in range(N):
                    if mask & (1 << d):
                        continue
                    for k in range(j + 1, L + 1):
                        nv = max(val, X[last][d][j] + P[d][k] - P[d][j])
                        nk = (mask | (1 << d), d, k)
                        if nk not in dp or nv < dp[nk][0]:
                            dp[nk] = (nv, key)
    key = min((k for k in dp if k[2] == L), key=lambda k: dp[k][0])
    assign = [0] * L
    while key is not None:
        prev = dp[key][1]
        s = prev[2] if prev else 0
        for l in range(s, key[2]):
            assign[l] = key[1]
        key = prev
    return assign


def metric(e):
    return e["latency"] if OBJECTIVE == "latency" else e["bottleneck"]


def plan_for(T, X, N, L, start):
    if OBJECTIVE == "latency":
        return plan_latency(T, X, N, L, start)
    return plan_throughput(T, X, N, L, start)


def decide(T, X, N, L, start):
    raw = plan_for(T, X, N, L, start)
    raw_eval = evaluate(raw, T, X, start)
    singles = [evaluate([i] * L, T, X, start) for i in range(N)]
    best_i = min(range(N), key=lambda i: metric(singles[i]))
    gain = 1 - metric(raw_eval) / metric(singles[best_i])
    final = raw if gain >= MIN_GAIN else [best_i] * L
    return raw, final, best_i, gain


def sweep_grid(T, X, N, L, start):
    slow = max(range(N), key=lambda i: sum(T[i]))
    base_link = min(range(len(LINK_MULTS)), key=lambda i: abs(LINK_MULTS[i] - 1))
    gain_grid = [[0.0] * len(LINK_MULTS) for _ in SPEED_MULTS]
    placements = []
    rows = []
    for si, sm in enumerate(SPEED_MULTS):
        T2 = [list(t) for t in T]
        T2[slow] = [t / sm for t in T[slow]]
        for li, lm in enumerate(LINK_MULTS):
            X2 = [[[v * lm for v in X[a][b]] for b in range(N)] for a in range(N)]
            raw, final, best_i, gain = decide(T2, X2, N, L, start)
            gain_grid[si][li] = gain * 100
            rows.append({"speed_mult": sm, "link_mult": lm, "gain_pct": round(gain * 100, 2),
                         "devices_used": len(set(final)), "split_chosen": len(set(final)) > 1})
            if li == base_link:
                placements.append(final)
    return slow, gain_grid, placements, rows


def run_sweep(T, X, N, L, start, names):
    slow, gain_grid, placements, rows = sweep_grid(T, X, N, L, start)
    print("\n=== SENSITIVITY SWEEP ===")
    print(f"hypothetical: '{names[slow]}' made faster by the speed multiplier, all link costs scaled by the link multiplier")
    print(f"cells below the {MIN_GAIN * 100:.0f}% gain threshold show 'single' (no split is used)")
    gdf = pd.DataFrame(gain_grid,
                       index=[f"{names[slow]} x{m}" for m in SPEED_MULTS],
                       columns=[f"link x{m}" for m in LINK_MULTS]).round(1)
    display(gdf)
    gdf.to_csv("sweep_gain_pct.csv")
    pd.DataFrame(rows).to_csv("sweep_results.csv", index=False)

    colors = [f"C{i}" for i in range(N)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    vmax = max(max(r) for r in gain_grid) or 1.0
    im = ax[0].imshow(gain_grid, aspect="auto", origin="lower", cmap="YlGnBu", vmin=0, vmax=vmax)
    for si in range(len(SPEED_MULTS)):
        for li in range(len(LINK_MULTS)):
            g = gain_grid[si][li]
            txt = "single" if g < MIN_GAIN * 100 else f"{g:.0f}%"
            ax[0].text(li, si, txt, ha="center", va="center", fontsize=9,
                       color="white" if g > 0.6 * vmax else "black")
    ax[0].set_xticks(range(len(LINK_MULTS)))
    ax[0].set_xticklabels([f"x{m}" for m in LINK_MULTS])
    ax[0].set_yticks(range(len(SPEED_MULTS)))
    ax[0].set_yticklabels([f"x{m}" for m in SPEED_MULTS])
    ax[0].set(xlabel="link cost multiplier (higher = slower link)",
              ylabel=f"{names[slow]} speed multiplier (higher = faster)",
              title="Predicted gain of the best split over the best single device")
    fig.colorbar(im, ax=ax[0], label="gain %")

    ax[1].imshow(placements, aspect="auto", origin="lower",
                 cmap=ListedColormap(colors), vmin=-0.5, vmax=N - 0.5)
    ax[1].set_xticks(range(L))
    ax[1].set_xticklabels([f"L{l + 1}" for l in range(L)])
    ax[1].set_yticks(range(len(SPEED_MULTS)))
    ax[1].set_yticklabels([f"x{m}" for m in SPEED_MULTS])
    ax[1].set(xlabel="layer", ylabel=f"{names[slow]} speed multiplier",
              title="Chosen placement at the measured link cost")
    ax[1].legend(handles=[Patch(color=colors[i], label=names[i]) for i in range(N)],
                 loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=N)
    fig.suptitle(f"Sensitivity of the partition ({OBJECTIVE} objective, batch {BATCH})", fontsize=14)
    fig.tight_layout()
    fig.savefig("sweep_plots.png", dpi=150)
    plt.show()
    print("saved: sweep_gain_pct.csv, sweep_results.csv, sweep_plots.png")


@torch.inference_mode()
def run_plan(assign, x, dev_units, start=0):
    def dev_of(i):
        return torch.device(SPECS[i]["torch"])

    old = torch.get_num_threads()
    cur = start
    t = x.to(dev_of(start))
    for l, d in enumerate(assign):
        if d != cur:
            ms, kind = link_cost(cur, d, ACT_BYTES[l])
            t = t.to(dev_of(d))
            if kind == "simulated":
                time.sleep(ms / 1000)
            cur = d
        if SPECS[d]["torch"] == "cpu":
            torch.set_num_threads(SPECS[d].get("threads") or old)
        t = dev_units[d][l](t)
    if cur != start:
        ms, kind = link_cost(cur, start, ACT_BYTES[len(assign)])
        t = t.to(dev_of(start))
        if kind == "simulated":
            time.sleep(ms / 1000)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.set_num_threads(old)
    return t


def measure_plan(assign, x, dev_units, warmup=5, iters=MEASURE_ITERS):
    for _ in range(warmup):
        run_plan(assign, x, dev_units)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_plan(assign, x, dev_units)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main():
    global SPECS, ACT_BYTES
    torch.manual_seed(SEED)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)

    units = build_units()
    L = len(units)

    x0 = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
    acts = [x0]
    with torch.inference_mode():
        for _, u in units:
            acts.append(u(acts[-1]))
    ACT_BYTES = [a.numel() * 4 for a in acts]

    SPECS = select_devices()
    N = len(SPECS)
    names = [s["name"] for s in SPECS]
    T, dev_units = profile_all(SPECS, units, acts)
    X = [[[link_cost(a, b, ACT_BYTES[k])[0] for k in range(L + 1)] for b in range(N)] for a in range(N)]
    start = 0

    seen, rows = set(), []
    for s in SPECS:
        if "profile_file" in s:
            kind = "external profile"
        elif s["torch"] in seen:
            kind = "virtual (same hardware)"
        else:
            kind = "real"
        seen.add(s["torch"])
        rows.append({"device": s["name"], "torch": s["torch"], "threads": s.get("threads") or "all", "kind": kind})
    print(f"=== DEVICES (objective: {OBJECTIVE}, batch {BATCH}) ===")
    display(pd.DataFrame(rows).set_index("device"))

    raw_assign, assign, best_i, gain = decide(T, X, N, L, start)
    plan = evaluate(assign, T, X, start)
    if gain >= MIN_GAIN:
        print(f"DECISION: split across devices, predicted gain {gain * 100:.1f}% (threshold {MIN_GAIN * 100:.0f}%)")
    else:
        print(f"DECISION: everything on {names[best_i]}, best split gains only {gain * 100:.1f}% "
              f"(threshold {MIN_GAIN * 100:.0f}%)")

    layer_rows = []
    for l, (name, _) in enumerate(units):
        r = {"layer": name, "input MB": round(ACT_BYTES[l] / 1e6, 3)}
        for i, n in enumerate(names):
            r[f"{n} ms"] = round(T[i][l], 3)
        r["fastest alone"] = names[min(range(N), key=lambda i: T[i][l])]
        r["best split"] = names[raw_assign[l]]
        r["assigned"] = names[assign[l]]
        layer_rows.append(r)
    ldf = pd.DataFrame(layer_rows).set_index("layer")
    print("=== PER-LAYER TIMES AND ASSIGNMENT ===")
    display(ldf)
    ldf.to_csv("layer_profile.csv")

    print("=== WHY EACH LAYER IS WHERE IT IS ===")
    for l, (name, _) in enumerate(units):
        c = assign[l]
        f = min(range(N), key=lambda i: T[i][l])
        times = ", ".join(f"{names[i]} {T[i][l]:.3f}" for i in range(N))
        if c == f:
            why = f"fastest device for this layer ({times} ms)"
        else:
            prev = assign[l - 1] if l > 0 else start
            nxt = assign[l + 1] if l < L - 1 else start
            extra = (X[prev][f][l] + X[f][nxt][l + 1]) - (X[prev][c][l] + X[c][nxt][l + 1])
            saving = T[c][l] - T[f][l]
            if OBJECTIVE == "latency":
                why = (f"{names[f]} is {saving:.3f} ms faster here, but moving it there adds "
                       f"{extra:.3f} ms of transfer ({times} ms)")
            else:
                why = (f"placed to balance pipeline stages; {names[f]} is {saving:.3f} ms faster here "
                       f"({times} ms)")
        print(f"{name:<24} -> {names[c]:<9} {why}")

    print("\n=== STAGES ===")
    srows = []
    for i, s in enumerate(plan["stages"]):
        srows.append({
            "stage": i + 1,
            "device": names[s["dev"]],
            "layers": f"L{s['start'] + 1}-L{s['end'] + 1}",
            "compute ms": round(s["compute"], 3),
            "transfer in ms": round(s["tx_in"], 3),
            "sent MB": round(s["tx_bytes"] / 1e6, 3),
            "stage total ms": round(s["total"], 3),
        })
    display(pd.DataFrame(srows).set_index("stage"))
    if plan["ret"] > 0:
        print(f"return transfer to {names[start]}: {plan['ret']:.3f} ms")

    configs = {f"all on {n}": [i] * L for i, n in enumerate(names)}
    configs["best split found"] = raw_assign
    all_exec = all(s["executable"] for s in SPECS)
    measured = {}
    if all_exec:
        print("\nvalidating by really running each configuration ...")
        ref = run_plan([0] * L, x0, dev_units, start)
        out = run_plan(raw_assign, x0, dev_units, start)
        diff = (ref.cpu() - out.cpu()).abs().max().item()
        print(f"split output matches single-device output: {diff < 1e-3} (max diff {diff:.2e})")
        for k, a in configs.items():
            measured[k] = measure_plan(a, x0, dev_units)
    else:
        print("\nskipping real execution: at least one device uses an external profile")

    crow = []
    for k, a in configs.items():
        e = evaluate(a, T, X, start)
        row = {"configuration": k,
               "predicted latency ms": round(e["latency"], 3),
               "predicted bottleneck ms": round(e["bottleneck"], 3),
               "predicted img/s (pipelined)": round(BATCH * 1000 / e["bottleneck"], 1)}
        if measured:
            row["measured latency ms"] = round(measured[k], 3)
            row["prediction error %"] = round((e["latency"] - measured[k]) / measured[k] * 100, 1)
        crow.append(row)
    cdf = pd.DataFrame(crow).set_index("configuration")
    print("\n=== BEST SPLIT vs SINGLE DEVICE ===")
    display(cdf)
    cdf.to_csv("partition_comparison.csv")

    col = "predicted latency ms" if OBJECTIVE == "latency" else "predicted bottleneck ms"
    singles_only = [c for c in cdf.index if c != "best split found"]
    best_single = min(singles_only, key=lambda c: cdf.loc[c, col])
    ratio = cdf.loc[best_single, col] / cdf.loc["best split found", col]
    print(f"predicted: best split is {ratio:.2f}x vs best single device ({best_single}) on {col}")
    if measured:
        mratio = measured[best_single] / measured["best split found"]
        print(f"measured: best split is {mratio:.2f}x vs {best_single} on end-to-end latency")

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    colors = [f"C{i}" for i in range(N)]
    xs = list(range(L))
    w = 0.8 / N

    for i in range(N):
        pos = [x + (i - (N - 1) / 2) * w for x in xs]
        ax[0, 0].bar(pos, T[i], w, color=colors[i], label=names[i],
                     edgecolor=["black" if assign[l] == i else "none" for l in xs],
                     linewidth=[2.5 if assign[l] == i else 0 for l in xs])
    ax[0, 0].set_xticks(xs)
    ax[0, 0].set_xticklabels([f"L{l + 1}" for l in xs])
    ax[0, 0].set(title="Time per layer on each device (black outline = chosen)", ylabel="ms")
    ax[0, 0].legend()
    ax[0, 0].grid(True, axis="y", alpha=0.3)

    t = 0.0
    for s in plan["stages"]:
        d = s["dev"]
        if s["tx_in"] > 0:
            ax[0, 1].barh(d, s["tx_in"], left=t, color="lightgray", edgecolor="black", hatch="//")
            t += s["tx_in"]
        for l in range(s["start"], s["end"] + 1):
            ax[0, 1].barh(d, T[d][l], left=t, color=colors[d], edgecolor="white")
            if T[d][l] > plan["latency"] * 0.02:
                ax[0, 1].text(t + T[d][l] / 2, d, f"L{l + 1}", ha="center", va="center", fontsize=8, color="white")
            t += T[d][l]
    if plan["ret"] > 0:
        ax[0, 1].barh(start, plan["ret"], left=t, color="lightgray", edgecolor="black", hatch="//")
    ax[0, 1].set_yticks(range(N))
    ax[0, 1].set_yticklabels(names)
    ax[0, 1].set(title="Timeline of the plan (hatched = transfer)", xlabel="ms")
    ax[0, 1].grid(True, axis="x", alpha=0.3)

    labels = list(configs.keys())
    pred = [cdf.loc[k, "predicted latency ms"] for k in labels]
    px = list(range(len(labels)))
    if measured:
        ax[1, 0].bar([p - 0.2 for p in px], pred, 0.4, label="predicted")
        ax[1, 0].bar([p + 0.2 for p in px], [measured[k] for k in labels], 0.4, label="measured")
        ax[1, 0].legend()
    else:
        ax[1, 0].bar(px, pred, 0.6)
    ax[1, 0].set_xticks(px)
    ax[1, 0].set_xticklabels(labels, rotation=20, ha="right")
    ax[1, 0].set(title="End-to-end latency: single device vs partitioned", ylabel="ms")
    ax[1, 0].grid(True, axis="y", alpha=0.3)

    cuts = {s["start"] for s in plan["stages"] if s["tx_in"] > 0}
    bar_colors = ["red" if k in cuts else "steelblue" for k in range(L + 1)]
    ax[1, 1].bar(range(L + 1), [b / 1e6 for b in ACT_BYTES], color=bar_colors)
    ax[1, 1].set_xticks(range(L + 1))
    ax[1, 1].set_xticklabels([f"into L{k + 1}" for k in range(L)] + ["final output"],
                             rotation=45, ha="right", fontsize=8)
    ax[1, 1].set(title="Data size entering each layer (red = where the plan transfers)", ylabel="MB")
    ax[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Layer partitioning across {N} devices ({OBJECTIVE} objective, batch {BATCH})", fontsize=14)
    fig.tight_layout()
    fig.savefig("partition_plots.png", dpi=150)
    plt.show()
    print("saved: layer_profile.csv, partition_comparison.csv, partition_plots.png, profile_<device>.json")

    if RUN_SWEEP and N >= 2:
        run_sweep(T, X, N, L, start, names)


if __name__ == "__main__":
    main()
