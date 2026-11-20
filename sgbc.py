"""SGBC-Net for 12-step freeway flow forecasting. Tensors use (batch, time, detector, channel)."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def aggregate(graph, x):
    return torch.einsum("ij,btjc->btic", graph, x)


def causal_conv(x, conv):
    b, t, n, c = x.shape
    signals = x.permute(0, 2, 3, 1).reshape(b*n, c, t)
    values = conv(F.pad(signals, (conv.kernel_size[0]-1, 0)))
    return values.reshape(b, n, c, t).permute(0, 3, 1, 2)


class DifferenceCorrection(nn.Module):
    """State-conditioned correction. Variants 1–3 correspond to b0, b5 and b6."""
    def __init__(self, channels, variant):
        super().__init__()
        self.variant = variant
        self.lin = nn.Linear(channels, channels)
        self.compress = nn.Linear(channels, channels // 2)
        self.expand = nn.Linear(channels // 2, channels)
        if variant == 3:
            self.gate_net = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                          nn.Linear(channels, channels), nn.Sigmoid())
        else:
            self.gate_net = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, graph, symmetric):
        h = aggregate(graph, self.lin(x)).relu()
        candidate = self.expand(self.compress(h).relu())
        gate = self.gate_net(h if self.variant == 1 else x)
        return h + self.alpha * ((1 - gate) * (candidate - h))


class GraphBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(channels, channels), nn.ReLU())
        self.compress = nn.Linear(channels, channels // 4)
        self.expand = nn.Linear(channels // 4, channels)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, graph, symmetric):
        context = self.expand(self.compress(aggregate(graph, x)).relu())
        return self.main(x) + self.alpha * context


class Separable(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.lin = nn.Linear(c, c)
        self.tconv = nn.Conv1d(c, c, 2)

    def forward(self, x, graph, symmetric):
        return causal_conv(self.lin(aggregate(symmetric, x)), self.tconv)


class TemporalAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.lin = nn.Linear(c, c)
        self.attn = nn.MultiheadAttention(c, 2, batch_first=True)
        self.norm = nn.LayerNorm(c)

    def forward(self, x, graph, symmetric):
        b, t, n, c = x.shape
        query = x.transpose(1, 2).reshape(b*n, t, c)
        values = self.lin(aggregate(symmetric, x)).transpose(1, 2).reshape(b*n, t, c)
        result, _ = self.attn(query, query, values)
        return self.norm(result.reshape(b, n, t, c).transpose(1, 2) + x)


class Adaptive(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        self.e1 = nn.Parameter(.01 * torch.randn(n, 10))
        self.e2 = nn.Parameter(.01 * torch.randn(n, 10))
        self.lin = nn.Linear(c, c)

    def forward(self, x, graph, symmetric):
        weights = (self.e1 @ self.e2.T).relu().softmax(1)
        return self.lin(aggregate(weights, x))


class GraphAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.norm = nn.LayerNorm(c)

    def forward(self, x, graph, symmetric):
        scores = (self.q(x) @ self.k(x).transpose(-2, -1)) * x.shape[-1]**-.5
        weights = scores.masked_fill(graph <= 0, -torch.inf).softmax(-1)
        return self.norm(weights @ self.v(x) + x)


class Joint(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(c,c), nn.Linear(c,c)
        self.tconv1, self.tconv2 = nn.Conv1d(c,c,2), nn.Conv1d(c,c,2)
        self.norm = nn.LayerNorm(c)

    def forward(self, x, graph, symmetric):
        h = causal_conv(self.lin1(aggregate(symmetric, x)), self.tconv1)
        h = causal_conv(self.lin2(aggregate(symmetric, h)), self.tconv2)
        return self.norm(h + x)


class Recurrent(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gru = nn.GRU(c, c, batch_first=True)

    def forward(self, x, graph, symmetric):
        b,t,n,c = x.shape
        h, _ = self.gru(x.transpose(1,2).reshape(b*n,t,c))
        return h.reshape(b,n,t,c).transpose(1,2)


class Block(nn.Module):
    def __init__(self, spatial, temporal, *, sequential=False):
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal
        self.sequential = sequential
        self.norm = nn.LayerNorm(64)

    def forward(self, x, graph, symmetric):
        s = self.spatial(x, graph, symmetric)
        if self.sequential:
            update = self.temporal(s, graph, symmetric)
        else:
            h = x if self.temporal is None else self.temporal(x,graph,symmetric)
            update = h * s.sigmoid()
        return self.norm(x + update)


class SGBCNet(nn.Module):
    def __init__(self, adjacency):
        super().__init__()
        a = torch.as_tensor(adjacency,dtype=torch.float32)
        n = a.size(0)
        if a.ndim != 2 or a.size(1) != n or not torch.isfinite(a).all() or (a < 0).any():
            raise ValueError("Expected a square nonnegative detector adjacency matrix")
        eye = torch.eye(n, device=a.device)
        raw = a + eye
        graph = (raw.double() / raw.sum(1,keepdim=True)).float()
        g = graph + eye
        scale = g.sum(1).clamp_min(1e-12).rsqrt()
        self.register_buffer("graph",graph,persistent=False)
        self.register_buffer("symmetric",scale[:,None]*g*scale[None,:],persistent=False)
        self.input_proj = nn.Linear(3,64)
        self.node = nn.Parameter(torch.empty(n,128))
        nn.init.xavier_uniform_(self.node)
        self.tod = nn.Embedding(288,64)
        self.dow = nn.Embedding(7,64)
        self.fuse = nn.Linear(320,64)
        self.blocks = nn.ModuleList([
            Block(DifferenceCorrection(64,1),None), Block(Separable(64),TemporalAttention(64)),
            Block(Adaptive(64,n),Separable(64)), Block(GraphAttention(64),None),
            Block(GraphBottleneck(64),Joint(64)), Block(DifferenceCorrection(64,2),GraphBottleneck(64),sequential=True),
            Block(DifferenceCorrection(64,3),Recurrent(64))])
        self.dropout = nn.Dropout(0.043564649850770354)
        self.out_proj = nn.Linear(12*64,12)

    def forward(self, x, time_of_day=None, day_of_week=None):
        b,t,n,_ = x.shape
        zeros = torch.zeros(b,t,dtype=torch.long,device=x.device)
        tod = self.tod(zeros if time_of_day is None else time_of_day.long())
        dow = self.dow(zeros if day_of_week is None else day_of_week.long())
        parts = [self.input_proj(x), self.node.expand(b,t,-1,-1),
                 tod.unsqueeze(2).expand(-1,-1,n,-1), dow.unsqueeze(2).expand(-1,-1,n,-1)]
        h = self.fuse(torch.cat(parts,-1))
        for block in self.blocks:
            h = block(h,self.graph,self.symmetric)
        return self.out_proj(self.dropout(h).transpose(1,2).reshape(b,n,-1)).transpose(1,2)

    @classmethod
    def from_checkpoint(cls, adjacency, checkpoint):
        """Load a distributed checkpoint with the supplied raw adjacency in training detector order."""
        model = cls(adjacency)
        original = torch.load(Path(checkpoint),map_location="cpu",weights_only=True)
        names = {"embed.node_emb.emb":"node", "embed.tod_emb.emb.weight":"tod.weight",
                 "embed.dow_emb.emb.weight":"dow.weight", "head.weight":"out_proj.weight", "head.bias":"out_proj.bias"}
        state = {names.get(k,k.removeprefix("embed.").replace(".inner.", ".")):v for k,v in original.items()}
        model.load_state_dict(state,strict=True)
        return model.eval()
