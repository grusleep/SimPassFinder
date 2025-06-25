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
        self.n_heads = 4
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU(args.relu)

        self.embed_category = nn.Embedding(101, self.emb_dim)
        self.embed_country = nn.Embedding(92, self.emb_dim)
        self.embed_sl = nn.Embedding(6, self.emb_dim)
        self.embed_url = nn.Embedding(128, self.emb_dim)

        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(self.emb_dim * 2, self.emb_dim)
    
        input_dim = self.emb_dim * 4 + 32
        output_dim = self.n_hidden
        heads = [self.n_heads] * (self.n_layers-1) + [1]
        self.gat_layers = nn.ModuleList()
        for i in range(self.n_layers):
            activation = None if i == self.n_layers - 1 else self.relu
            self.gat_layers.append(GATConv(
                                            input_dim, 
                                            output_dim, 
                                            num_heads=heads[i], 
                                            feat_drop=args.dropout, 
                                            attn_drop=args.dropout, 
                                            negative_slope=args.relu, 
                                            residual=True, 
                                            activation=activation, 
                                            allow_zero_in_degree=True
                                           ))
            input_dim = output_dim * heads[i]
            
        self.fc_out = nn.Linear(self.n_hidden, 2)
        
    
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
        sec_emb = self.embed_sl(inputs_sl).squeeze(1)
        ip_emb = inputs_ip.float()
        
        h = torch.cat([h, cat_emb, country_emb, sec_emb, ip_emb], dim=1)
        
        attn = None
        for i, layer in enumerate(self.gat_layers):
            if i == self.n_layers - 1:
                h, attn = layer(blocks[i][('site','sim','site')], h, get_attention=True)
                h = h.squeeze(1)
            else:
                h = layer(blocks[i][('site','sim','site')]  , h)
                h = h.flatten(1)
                h = self.relu(h)
                h = self.dropout(h)
        
        src, dst = edge_sub.edges(etype='sim', order='eid')
        h_src = h[src]
        h_dst = h[dst]
        e_feat = h_src * h_dst
        scores = self.fc_out(e_feat)
        
        return scores, attn