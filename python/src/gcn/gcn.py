import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import dgl
from dgl.nn import GraphConv, HeteroGraphConv

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.emb_dim = args.embed_size
        self.hidden_size = args.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU(args.relu)
        self.num_layers = args.gnn_depth
        self.with_rules = args.feature == "with_rules"

        # Embedding layers for categorical features
        self.embed_category = nn.Embedding(101, self.emb_dim)
        self.embed_country = nn.Embedding(92, self.emb_dim)
        self.embed_sl = nn.Embedding(6, self.emb_dim)
        self.embed_url = nn.Embedding(128, self.emb_dim)
        if self.with_rules:
            self.embed_rules = nn.Embedding(9, self.emb_dim)

        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(self.emb_dim * 2, self.emb_dim)

        self.input_dim = self.emb_dim * 4  # +4 for float IP
        if self.with_rules:
            self.input_dim += self.emb_dim

        # GCN layers with HeteroGraphConv (edge type: 'sim')
        self.gcn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.input_dim if i == 0 else self.hidden_size
            self.gcn_layers.append(
                HeteroGraphConv(
                    {('site', 'sim', 'site'): GraphConv(in_dim, self.hidden_size, allow_zero_in_degree=True),
                     ('site', 'user', 'site'): GraphConv(in_dim, self.hidden_size, allow_zero_in_degree=True)},
                    aggregate='sum'  # can also be 'mean' or 'max'
                )
            )

        # Final edge-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )

    def forward(self, edge_sub, blocks, batch_inputs):
        if self.with_rules:
            inputs_s, inputs_sm, inputs_c, inputs_co, inputs_sl, inputs_r = batch_inputs
        else:
            inputs_s, inputs_sm, inputs_c, inputs_co, inputs_sl = batch_inputs

        # 1. Node feature construction
        lengths = inputs_sm.sum(dim=1)
        url_emb = self.embed_url(inputs_s)
        packed = pack_padded_sequence(url_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_url = torch.cat([h_n[0], h_n[1]], dim=1)
        h_url = self.fc_lstm(h_url)
        h_url = self.relu(h_url)
        h_url = self.dropout(h_url)

        h_cat = self.embed_category(inputs_c).squeeze(1)
        h_country = self.embed_country(inputs_co).squeeze(1)
        h_sec = self.embed_sl(inputs_sl).squeeze(1)
        if self.with_rules:
            h_rules = self.embed_rules(inputs_r.long())

        if self.with_rules:
            h = torch.cat([h_url, h_cat, h_country, h_sec, h_rules], dim=1)
        else:
            h = torch.cat([h_url, h_cat, h_country, h_sec], dim=1)

        # 2. GNN propagation using HeteroGraphConv
        h_dict = {'site': h}
        for i in range(self.num_layers):
            h_dict = self.gcn_layers[i](blocks[i], h_dict)
            h = h_dict['site']
            h = self.relu(h)
            h = self.dropout(h)
            h_dict['site'] = h


        # 3. Edge-level prediction
        src, dst = edge_sub.edges(etype='sim', order='eid')
        h_src = h[src]
        h_dst = h[dst]
        edge_feat = torch.cat([h_src, h_dst], dim=1)

        out = self.classifier(edge_feat)
        return out, None
