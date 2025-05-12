import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dgl.nn import GATConv



class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.emb_dim = args.embed_size
        self.n_hidden = args.hidden_size
        self.n_layers = args.gnn_depth
        self.n_heads = 1
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU(args.relu)

        self.embed_category = nn.Embedding(26, self.emb_dim)
        self.embed_country = nn.Embedding(59, self.emb_dim)
        self.embed_security_level = nn.Embedding(6, self.emb_dim)
        self.embed_url = nn.Embedding(128, self.emb_dim)

        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(self.emb_dim * 2, self.emb_dim)
        
        heads = [self.n_heads] * [self.n_layers - 1] + [1]
        input_dim = 4 * self.emb_dim + self.ip_dim
        self.gat_convs = nn.ModuleList()
        for l in range(self.n_layers):
            in_dim = input_dim if l == 0 else self.n_hidden * heads[l-1]
            out_dim = self.n_hidden if l != self.n_layers - 1 else self.emb_dim
            activation = self.relu if l != self.n_layers - 1 else None
            self.gat_convs.append(
                GATConv(
                    in_dim, out_dim, heads[l],
                    feat_drop=args.dropout, attn_drop=args.dropout,
                    activation=activation
                )
            )
        
    
    def forward(self, edge_sub, blocks, inputs_s, inputs_sm, inputs_c, inputs_co, inputs_sl, inputs_ip):
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
        sec_emb = self.embed_security_level(inputs_sl).squeeze(1)
        ip_emb = inputs_ip.float()
        
        h = torch.cat([h, cat_emb, country_emb, sec_emb, ip_emb], dim=1)
        for l, conv in enumerate(self.gat_convs):
            h_new = conv(blocks[l], h_feat)
            if l != self.n_layers - 1:
                h_feat = h_new.flatten(1)
                h_feat = self.dropout(h_feat)
            else:
                h_feat = h_new.squeeze(1)

        src, dst = edge_sub.edges(order='eid')
        src_h = h_feat[src]
        dst_h = h_feat[dst]
        scores = torch.sum(src_h * dst_h, dim=1)
        return scores, None