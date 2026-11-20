"""Evaluate a frozen SGBC-Net from the public PeMS flow array and distance edge list."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from sgbc import SGBCNet

HERE = Path(__file__).resolve().parent


def load_inputs(dataset, data_dir):
    path = data_dir / (dataset.lower()+".npz")
    expected = {"PeMS04":"95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a",
                "PeMS08":"e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075"}
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected[dataset]:
        raise ValueError(f"Unexpected raw flow file for {dataset}")
    with np.load(path) as inp:
        raw = inp["data"].astype(np.float32)
    n = raw.shape[1]
    distances = np.full((n,n),np.inf)
    for src,dst,length in np.genfromtxt(data_dir/"distance.csv",delimiter=",",skip_header=1):
        i,j = int(src),int(dst)
        distances[i,j] = length
        distances[j,i] = min(distances[j,i],length)
    sigma = distances[np.isfinite(distances)].std()
    graph = np.exp(-(distances/sigma)**2)
    graph[graph<.1] = 0
    return raw, graph.astype(np.float32)


def test_batches(raw, mean, std, batch):
    count = len(raw)-23
    start = int(.6*count)+int(.2*count)
    for left in range(start,count,batch):
        t = np.arange(left,min(left+batch,count))[:,None]+np.arange(12)
        x = raw[t].copy()
        x[...,0] = (x[...,0]-mean)/std
        y = (raw[t+12,:,0]-mean)/std
        yield x, t%288, (t//288)%7, y


def evaluate(dataset,data_dir,device,batch):
    raw,graph = load_inputs(dataset,data_dir)
    mean,std = map(float,np.load(HERE/"checkpoints"/(dataset+"_scaler.npy")))
    model = SGBCNet.from_checkpoint(graph,HERE/"checkpoints"/(dataset+".pt")).to(device)
    totals = np.zeros((12,4),np.float64)
    with torch.inference_mode():
        for x,tod,dow,y in test_batches(raw,mean,std,batch):
            x,tod,dow = [torch.from_numpy(v).to(device) for v in (x,tod,dow)]
            forecast = model(x,tod,dow).cpu()*std+mean
            truth = torch.from_numpy(y)*std+mean
            valid = ~torch.isclose(truth,torch.zeros((),dtype=truth.dtype))
            delta = (forecast-truth).double()
            if not torch.isfinite(delta).all():
                raise ValueError("Non-finite forecast")
            for j,val in enumerate((delta.abs(),delta.square(),delta.abs()/truth.double(),torch.ones_like(delta))):
                totals[:,j] += torch.where(valid,val,0).sum((0,2)).numpy()
    for lead,sums in [("pooled",totals.sum(0)),*[(str(5*(i+1)),s) for i,s in enumerate(totals)]]:
        a,sq,ape,count = sums
        yield [dataset,lead,a/count,np.sqrt(sq/count),100*ape/count]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",choices=["PeMS04","PeMS08"])
    parser.add_argument("data_dir",type=Path)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    if args.batch<1:parser.error("Batch size must be positive")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    import sys
    writer=csv.writer(sys.stdout)
    writer.writerow(["dataset","lead_minutes","mae","rmse","mape_percent"])
    writer.writerows(evaluate(args.dataset,args.data_dir,args.device,args.batch))
