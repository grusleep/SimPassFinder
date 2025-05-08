import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.emb_dim = args.embed_size
        self.n_hidden = args.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU(args.relu)

        self.embed_category = nn.Embedding(26, self.emb_dim)
        self.embed_country = nn.Embedding(59, self.emb_dim)
        self.embed_security_level = nn.Embedding(6, self.emb_dim)
        self.embed_url = nn.Embedding(128, self.emb_dim)

        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(self.emb_dim * 2, self.emb_dim)
    
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
        
        