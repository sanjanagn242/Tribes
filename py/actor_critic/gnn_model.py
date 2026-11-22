"""
Actor-critic graph network for Tribes.

Input (one board size per batch, so the grid edges are shared by every graph in the batch):
    x           [B, N, F]   node (tile) features
    edge_attr   [B, E, Fe]  edge features (8-neighbour grid, edge list from selfplay_data.edge_index)
    u           [B, G]      global (tribe-level) features
    actions     [B, A, Fa]  action features (ActionEncoder), padded to the longest action list
    action_mask [B, A]      True for real actions, False for padding

Output:
    logits [B, A]  one score per action (padding = -inf); softmax over them gives the actor's priors
    value  [B]     critic's estimate in [-1, 1] for the encoded player

Architecture:
    1. Embed nodes, edges and globals to `hidden` dims.
    2. `layers` rounds of message passing: each edge (s -> t) builds a message from (h_s, h_t, e_st);
       each node averages its incoming messages and updates itself with them and the global vector;
       the global vector is updated from the mean node embedding. Residual + LayerNorm everywhere.
    3. Graph embedding = MLP(mean(h), max(h), g)   ->  value head.
    4. Each action is scored from (h[src tile], h[target tile], action features, graph embedding), so the policy
       sees the full context of the tiles an action touches. Missing tiles (node id -1) contribute zeros.

Pure PyTorch (no torch_geometric layers), so it is easy to export later.
"""

import math

import torch
import torch.nn as nn

from selfplay_data import edge_index as grid_edge_index


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_out))


class GraphLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = mlp(3 * hidden, hidden, hidden)
        self.update = mlp(3 * hidden, hidden, hidden)
        self.glob = mlp(2 * hidden, hidden, hidden)
        self.norm_h = nn.LayerNorm(hidden)
        self.norm_g = nn.LayerNorm(hidden)

    def forward(self, h, e, g, src, tgt, inv_deg):
        # h [B,N,H], e [B,E,H], g [B,H]; src/tgt [E]; inv_deg [N]
        m = self.message(torch.cat([h[:, src], h[:, tgt], e], dim=-1))            # [B,E,H]
        agg = torch.zeros_like(h).index_add(1, tgt, m) * inv_deg[None, :, None]   # mean over incoming edges
        h = self.norm_h(h + self.update(torch.cat([h, agg, g[:, None, :].expand_as(h)], dim=-1)))
        g = self.norm_g(g + self.glob(torch.cat([g, h.mean(dim=1)], dim=-1)))
        return h, g


class TribesGNN(nn.Module):
    def __init__(self, node_features, edge_features, global_features, action_features,
                 src_col, tgt_col, hidden=128, layers=4):
        super().__init__()
        self.config = dict(node_features=node_features, edge_features=edge_features,
                           global_features=global_features, action_features=action_features,
                           src_col=src_col, tgt_col=tgt_col, hidden=hidden, layers=layers)
        self.src_col, self.tgt_col, self.hidden = src_col, tgt_col, hidden

        # Action columns fed to the MLP: everything except the two raw node-id columns.
        keep = [c for c in range(action_features) if c not in (src_col, tgt_col)]
        self.register_buffer("action_keep", torch.tensor(keep, dtype=torch.long), persistent=False)

        self.node_in = mlp(node_features, hidden, hidden)
        self.edge_in = mlp(edge_features, hidden, hidden)
        self.glob_in = mlp(global_features, hidden, hidden)
        self.layers = nn.ModuleList([GraphLayer(hidden) for _ in range(layers)])

        self.graph_out = mlp(3 * hidden, hidden, hidden)
        self.value_head = mlp(hidden, hidden, 1)
        self.action_in = mlp(len(keep), hidden, hidden)
        self.policy_head = nn.Sequential(nn.Linear(4 * hidden, 2 * hidden), nn.SiLU(),
                                         nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self._graphs = {}

    def _graph(self, n, device):
        """Edge list and 1/in-degree for a board with n nodes (cached per size and device)."""
        key = (n, str(device))
        if key not in self._graphs:
            size = int(round(math.sqrt(n)))
            assert size * size == n, f"{n} nodes is not a square board"
            ei = torch.from_numpy(grid_edge_index(size)).to(device)
            deg = torch.bincount(ei[1], minlength=n).clamp(min=1).float()
            self._graphs[key] = (ei[0], ei[1], 1.0 / deg)
        return self._graphs[key]

    def _gather_nodes(self, h, ids):
        """h [B,N,H], ids [B,A] float node ids (-1 = none) -> [B,A,H], zeros where id < 0."""
        ids = ids.long()
        present = (ids >= 0).unsqueeze(-1).to(h.dtype)
        idx = ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.shape[-1])
        return torch.gather(h, 1, idx) * present

    def forward(self, x, edge_attr, u, actions, action_mask):
        src, tgt, inv_deg = self._graph(x.shape[1], x.device)
        h, e, g = self.node_in(x), self.edge_in(edge_attr), self.glob_in(u)
        for layer in self.layers:
            h, g = layer(h, e, g, src, tgt, inv_deg)

        graph = self.graph_out(torch.cat([h.mean(dim=1), h.amax(dim=1), g], dim=-1))   # [B,H]
        value = torch.tanh(self.value_head(graph)).squeeze(-1)

        h_src = self._gather_nodes(h, actions[..., self.src_col])
        h_tgt = self._gather_nodes(h, actions[..., self.tgt_col])
        a = self.action_in(actions.index_select(-1, self.action_keep))
        ctx = graph[:, None, :].expand(-1, actions.shape[1], -1)
        logits = self.policy_head(torch.cat([h_src, h_tgt, a, ctx], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        return logits, value
