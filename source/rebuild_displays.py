"""Build the two panels of Table I and Figures 5 and 7 from archived test statistics."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "reproduction"
FIGURES = HERE / "figures"
NETWORKS = ("PeMS04", "PeMS08")
BLUE, ORANGE, GRAY = "#246780", "#a55424", "#c5ced1"


def read(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def table():
    pooled = read("comparison.csv")
    step_rows = read("horizon_records.csv")
    columns = ("mae", "rmse", "mape_percent")
    out = []
    for row in pooled:
        for ds in NETWORKS:
            key = ds.lower()
            out.append(["all_12_horizons", row["model_id"], row["model"], ds] +
                       [float(row[key + "_" + field]) for field in columns])
            steps = [r for r in step_rows if r["dataset"] == ds and r["model_id"] == row["model_id"]]
            assert [int(r["minutes"]) for r in steps] == list(range(5, 61, 5))
            assert all(int(r["step"])*5 == int(r["minutes"]) for r in steps)
            last = steps[-1]
            out.append(["60_min", row["model_id"], row["model"], ds, float(last["mae"]),
                        float(last["rmse"]), 100 * float(last["mape"])])
    assert len(out) == 52 and len(step_rows) == 312
    with (DATA / "table_I.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["scope", "model_id", "model", "dataset", *columns])
        w.writerows(out)


def export(fig, number):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"Figure_{number:02d}.{ext}", dpi=420)
    plt.close(fig)


def clean(ax):
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="x", color="#e3e7e8", linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def distance():
    rows = read("distance_counts.csv")
    # The same number of selections and candidates per receiver makes these pooled
    # shares equal to the mean of the corresponding detector-level shares.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.70))
    fig.subplots_adjust(left=.135, right=.985, top=.76, bottom=.19, wspace=.33)
    for col, (ax, ds) in enumerate(zip(axes, NETWORKS)):
        r = [v for v in rows if v["dataset"] == ds]
        assert len(r) == 4
        clean(ax)
        for j, (numerator, denominator, color, label) in enumerate([
                ("selected_count", "selection_total", BLUE, "Selected sources"),
                ("candidate_count", "candidate_total", GRAY, "Available sources")]):
            vals = np.array([100*int(v[numerator])/int(v[denominator]) for v in r])
            np.testing.assert_allclose(vals.sum(), 100)
            y = np.arange(4) + (j-.5)*.32
            ax.barh(y, vals, .28, color=color, label=label)
            for i, (yy, val) in enumerate(zip(y, vals)):
                if not (ds == "PeMS08" and i == 3):
                    ax.text(val+1.4, yy, f"{val:.1f}", va="center", fontsize=7.3)
        if ds == "PeMS08":
            ax.text(1.5, 3, "No disconnected pairs", va="center", color="#50595b", fontsize=8)
        ax.set_yticks(range(4), ["1 link", "2–3 links", "4+ links", "Disconnected"])
        ax.set(xlim=(0,102), ylim=(3.45,-.48), xticks=[0,25,50,75,100], xlabel="Share of sources (%)")
        ax.set_title(f"({chr(97+col)})  {ds}", loc="left", fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    export(fig, 5)


def gates():
    summary = read("correction_distributions.csv")
    estimates = read("gates.csv")
    ops = ["b0_SBDC", "b5_SBDC", "b6_SBDC", "b4_AGB", "b5_AGB"]
    modes = ["node_fixed_all","shared_fixed_all","node_fixed_b0","node_fixed_b5","node_fixed_b6"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.65))
    fig.subplots_adjust(left=.17, right=.985, bottom=.105, top=.87, wspace=.63, hspace=.74)
    for col, ds in enumerate(NETWORKS):
        top, bottom = axs[:, col]
        for ax in (top,bottom):
            clean(ax)
            ax.set_ylim(4.55,-.54)
        for i, op in enumerate(ops):
            for dim, color, mark, offset in [("detector",BLUE,"o",-.14),("window",ORANGE,"^",.14)]:
                r, = [v for v in summary if (v["dataset"],v["dimension"],v["operator"]) == (ds,dim,op)]
                lo,med,hi = [float(r[k]) for k in ("q1","median","q3")]
                top.plot([lo,hi],[i+offset]*2, color=color, lw=2)
                top.plot(med, i+offset, marker=mark, color=color, ms=4)
        top.set(xlim=(-.10,2.85),xticks=[0,1,2], xlabel="Relative correction norm")
        top.set_yticks(range(5),[s.replace("_"," · ") for s in ops])
        for i,mode in enumerate(modes):
            r, = [v for v in estimates if (v["dataset"],v["weighting"],v["block_origins"],v["metric"],v["mode"]) ==
                  (ds,"original","288","MAE",mode)]
            lo,med,hi = [float(r[k]) for k in ("lower","estimate","upper")]
            bottom.plot([lo,hi],[i,i], lw=1.7, color=BLUE)
            bottom.plot(med,i,marker="D" if i<2 else "o",color=BLUE,ms=4.6 if i<2 else 3.8)
        bottom.axvline(0,lw=.7,color="#697577")
        bottom.axhline(1.5,color="#acb7ba",lw=.6,ls=":")
        bottom.set(xlim=(-.035,.86),xticks=[0,.2,.4,.6,.8],xlabel="Increase in MAE (flow units)")
        bottom.set_yticks(range(5),["All · per detector","All · shared","b0 only","b5 only","b6 only"])
        for ax, k in ((top,col),(bottom,col+2)):
            ax.set_title(f"({chr(97+k)})  {ds}",loc="left",fontweight="bold")
    handles=[Line2D([],[],color=c,marker=m,ls="-",lw=2,markersize=4,label=t) for c,m,t in [
        (BLUE,"o","Across detectors"),(ORANGE,"^","Across input windows")]]
    fig.legend(handles=handles,loc="upper center",ncol=2,frameon=False)
    export(fig,7)


if __name__ == "__main__":
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],
                         "font.size":9,"axes.linewidth":.65,"pdf.fonttype":42})
    table()
    distance()
    gates()
