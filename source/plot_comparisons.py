"""Rebuild the three comparative figures. Run at any location in the repository or archive."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
HIST = NEW = HERE / "reproduction/inputs"
OUT = HERE / "figures"
DATA = HERE / "reproduction"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
NETWORKS = ["PeMS04", "PeMS08"]
STYLE = {"STID": ("#AD5A26", "o", "-"), "AGCRN": ("#755590", "s", "--"), "STLAformer": ("#176B78", "^", "-.")}
GROUPS = ["0–1", "2", "3", "4+"]

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def emit(name, rows):
    with (DATA / name).open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def group(degree):
    return GROUPS[min(max(degree-1,0),3)]

def finish(fig, axes, number):
    handles = [plt.Line2D([],[],color=c,marker=m,linestyle=s,markersize=4,label=n) for n,(c,m,s) in STYLE.items()]
    fig.legend(handles=handles,loc="upper center",ncols=3,frameon=False)
    for i,ax in enumerate(axes.flat):
        ax.set_title("("+chr(97+i)+")  "+NETWORKS[i%2],loc="left",fontweight="bold")
        ax.grid(axis="y",color="#dddddd",lw=.45)
        ax.spines[["right","top"]].set_visible(False)
    for ext in ("png","pdf"):
        fig.savefig(OUT / f"Figure_{number:02d}.{ext}",dpi=430)
    plt.close(fig)

def panels(rows):
    fig,axes=plt.subplots(rows,2,squeeze=False,figsize=(6.65,4.15 if rows==2 else 2.60))
    fig.subplots_adjust(left=.115,right=.987,top=.865 if rows==2 else .76,bottom=.106 if rows==2 else .21,wspace=.39,hspace=.64)
    return fig,axes

horizon = {}
for r in read(HIST/"relative_horizon_gains.csv"):
    if r["reference"] in STYLE:
        horizon[r["dataset"].lower(),r["reference"],r["metric"].lower(),int(r["minutes"])] = float(r["relative_reduction_percent"])
for r in read(NEW/"horizons.csv"):
    k=(r["dataset"].lower(),"STID",r["metric"],int(r["index"]))
    f=float(r["sgbc"])
    assert np.isclose(horizon[k],100*(1-f/float(r["stid"])))
    horizon[k[0],"STLAformer",k[2],k[3]]=100*(1-f/float(r["stlaformer"]))
emit("figure03_horizon.csv",[dict(dataset=d,reference=r,metric=m,minutes=t,reduction_pct=v) for (d,r,m,t),v in sorted(horizon.items())])
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],"font.size":8.5,"pdf.fonttype":42,"axes.linewidth":.65})
fig,axes=panels(2)
for col,ds in enumerate(NETWORKS):
    for row,metric in enumerate(("mae","rmse")):
        ax=axes[row,col]
        ax.axhline(0,lw=.7,color="#555")
        for ref,(color,marker,style) in STYLE.items():
            x=list(range(5,61,5))
            ax.plot(x,[horizon[ds.lower(),ref,metric,t] for t in x],color=color,marker=marker,ls=style,ms=3,lw=1.15)
        ax.set(xlabel="Forecast horizon (min)",ylabel=f"{metric.upper()} reduction (%)",xticks=[5,15,30,45,60])
finish(fig,axes,3)

paired={}
for r in read(HIST/"paired_nodes.csv"):
    if r["reference"] in STYLE:
        paired[r["dataset"],r["reference"],int(r["node"])]=(int(r["degree"]),float(r["delta_mae"]))
for r in read(NEW/"detectors.csv"):
    if r["metric"]!="mae": continue
    ds,node=r["dataset"].lower(),int(r["index"])
    degree,old=paired[ds,"STID",node]
    assert np.isclose(old,float(r["stid"])-float(r["sgbc"]))
    paired[ds,"STLAformer",node]=(degree,float(r["stlaformer"])-float(r["sgbc"]))
emit("figure04_detectors.csv",[dict(dataset=d,reference=r,detector=n,degree=g,difference=v) for (d,r,n),(g,v) in sorted(paired.items())])
fig,axes=panels(2)
fig.subplots_adjust(bottom=.15)
degree_rows=[]
for col,(ds,n) in enumerate(zip(NETWORKS,(307,170))):
    upper,lower=axes[:,col]
    upper.axvline(0,color="#555",lw=.7)
    lower.axhline(0,color="#555",lw=.7)
    for j,(ref,(color,marker,style)) in enumerate(STYLE.items()):
        degree,delta=zip(*(paired[ds.lower(),ref,node] for node in range(n)))
        upper.step(np.sort(delta),np.arange(1,n+1)/n,where="post",ls=style,color=color,lw=1.35)
        ns=[]
        for i,label in enumerate(GROUPS):
            v=[e for deg,e in zip(degree,delta) if group(deg)==label]
            q1,med,q3=np.quantile(v,[.25,.5,.75])
            degree_rows.append(dict(dataset=ds,reference=ref,degree_group=label,n=len(v),q25=q1,median=med,q75=q3))
            lower.errorbar(i+(j-1)*.16,med,yerr=[[med-q1],[q3-med]],color=color,fmt=marker,ms=3,capsize=2,lw=1)
            ns.append(len(v))
    upper.set(xscale="symlog",xlabel="Reference MAE − SGBC-Net MAE",ylabel="Cumulative proportion",ylim=(0,1.03))
    upper.set_xscale("symlog",linthresh=1)
    upper.set_xticks([-10,-1,0,1,10],["−10","−1","0","1","10"])
    lower.set(xlabel="Number of directly connected neighbors",ylabel="MAE difference (flow units)")
    lower.set_xticks(range(4),[f"{label}\n(n={n})" for label,n in zip(GROUPS,ns)])
emit("figure04_degree_groups.csv",degree_rows)
finish(fig,axes,4)

values={}
for r in read(HIST/"scenario_effects.csv"):
    if r["reference"] not in STYLE or (r["scenario"],r["metric"],r["weighting"],float(r["block_hours"]))!=("variation","MAE","all_nodes",24):continue
    values[r["dataset"],r["reference"],int(r["quartile"])]=[float(r[k]) for k in ("relative_percent","relative_lower","relative_upper")]
for r in read(NEW/"variation.csv"):
    if (r["weighting"],r["hours"])!=("all_nodes","24"): continue
    key=r["dataset"].lower(),r["reference"],int(r["quartile"])
    v=[float(r[k]) for k in ("reduction","lower","upper")]
    if key in values: np.testing.assert_allclose(v,values[key],atol=1e-6)
    values[key]=v
emit("figure06_variation.csv",[dict(dataset=d,reference=r,quartile=q,reduction_pct=v[0],low=v[1],high=v[2]) for (d,r,q),v in sorted(values.items())])
fig,axes=panels(1)
for ax,ds in zip(axes.flat,NETWORKS):
    ax.axhline(0,lw=.7,color="#555")
    for j,(ref,(color,marker,style)) in enumerate(STYLE.items()):
        vals=np.array([values[ds.lower(),ref,q] for q in range(1,5)])
        v,lo,hi=vals.T
        ax.errorbar(np.arange(1,5)+(j-1)*.075,v,yerr=[v-lo,hi-v],color=color,marker=marker,ls=style,ms=3,capsize=2,lw=1.15)
    ax.set(xlabel="Recent flow variation (low to high)",ylabel="MAE reduction (%)",xlim=(.7,4.3))
    ax.set_xticks(range(1,5),["Q1","Q2","Q3","Q4"])
finish(fig,axes,6)
