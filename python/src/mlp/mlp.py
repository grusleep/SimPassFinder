import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.emb_dim = args.embed_size
        self.n_hidden = args.hidden_size
        self.feature = args.feature
        self.with_rules = args.feature == "with_reuls"
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU(args.relu)

        self.embed_category = nn.Embedding(101, self.emb_dim)
        self.embed_country = nn.Embedding(92, self.emb_dim)
        self.embed_sl = nn.Embedding(6, self.emb_dim)
        self.embed_url = nn.Embedding(128, self.emb_dim)
        if self.with_rules:
            self.embed_rules = nn.Linear(9, self.emb_dim)

        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(self.emb_dim * 2, self.emb_dim)

        input_dim = self.emb_dim * 4 + 32
        if self.with_rules:
            input_dim += self.emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim*2, self.n_hidden),
            nn.LeakyReLU(args.relu),
            # nn.Dropout(args.dropout),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(args.relu),
            nn.Dropout(args.dropout),
            nn.Linear(self.n_hidden, 2)
        )


    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.fc(torch.cat([h_u, h_v], 1))
        score = self.relu(score)
        score = self.dropout(score)
        score = self.fc_out(score)
        attn = torch.cat([edges.src['attn'].squeeze(-1).unsqueeze(1), edges.src['attn'].squeeze(-1).unsqueeze(1)], dim=1)
        return {'score': score, 'attn': attn}
    
    
    def forward(self, edge_sub, blocks, batch_inputs):
        if self.with_rules:
            inputs_s, inputs_sm, inputs_c, inputs_co, inputs_sl, inputs_ip, inputs_r = batch_inputs
        else:
            inputs_s, inputs_sm, inputs_c, inputs_co, inputs_sl, inputs_ip = batch_inputs
        lengths = inputs_sm.sum(dim=1)
        url_emb = self.embed_url(inputs_s)
        packed = pack_padded_sequence(url_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        h = self.fc_lstm(h)
        h = self.relu(h)
        h = self.dropout(h)

        cat_emb = self.embed_category(inputs_c).squeeze(1)
        country_emb = self.embed_country(inputs_co).squeeze(1)
        sec_emb = self.embed_sl(inputs_sl).squeeze(1)
        ip_emb = inputs_ip.float()
        if self.with_rules:
            rules_emb = self.embed_rules(inputs_r.float())
            node_feat = torch.cat([h, cat_emb, country_emb, sec_emb, ip_emb, rules_emb], dim=1)
        else:
            node_feat = torch.cat([h, cat_emb, country_emb, sec_emb, ip_emb], dim=1)

        src, dst = edge_sub.edges(etype='sim',order='eid')
        edge_feat = torch.cat([node_feat[src], node_feat[dst]], dim=1)

        out = self.mlp(edge_feat)
        return out, None
