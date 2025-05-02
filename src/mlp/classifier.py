import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.emb_dim = args.embed_size
        self.n_hidden = args.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU(args.relu)

        # Embedding layers for categorical features
        self.embed_category = nn.Embedding(26, self.emb_dim)
        self.embed_country = nn.Embedding(59, self.emb_dim)
        self.embed_security_level = nn.Embedding(6, self.emb_dim)
        self.embed_url = nn.Embedding(128, self.emb_dim)

        # LSTM to encode URL sequences
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(self.emb_dim * 2, self.emb_dim)

        # MLP classifier head
        # Input dim: URL embedding + 3 categorical embeds + IP bits
        mlp_input_dim = self.emb_dim * 4 + 32
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.n_hidden),
            nn.LeakyReLU(args.relu),
            nn.Dropout(args.dropout),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(args.relu),
            nn.Dropout(args.dropout),
            nn.Linear(self.n_hidden, 2)
        )

    def forward(self, inputs_s, inputs_sm, inputs_c, inputs_co, inputs_sl, inputs_ip):
        # URL embedding and LSTM encoding
        # inputs_s: [batch_size, seq_len], inputs_sm: [batch_size, seq_len] mask
        lengths = inputs_sm.sum(dim=1)
        url_emb = self.embed_url(inputs_s)
        packed = pack_padded_sequence(url_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: [2, batch_size, emb_dim]
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        h = self.fc_lstm(h)
        h = self.relu(h)
        h = self.dropout(h)

        # Other feature embeddings
        cat_emb = self.embed_category(inputs_c).squeeze(1)
        country_emb = self.embed_country(inputs_co).squeeze(1)
        sec_emb = self.embed_security_level(inputs_sl).squeeze(1)
        ip_emb = inputs_ip.float()

        # Concatenate all features
        x = torch.cat([h, cat_emb, country_emb, sec_emb, ip_emb], dim=1)
        out = self.mlp(x)
        return out
