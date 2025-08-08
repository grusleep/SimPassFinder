import numpy as np
from dgl.nn.pytorch import Sequential
from .gnn import SAGEConvN

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class GraphSAGE(nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.emb_dim = args.embed_size
        self.n_hidden = args.hidden_size
        self.with_rules = args.feature == "with_rules"
        
        self.agg_type = args.agg_type
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU(args.relu)
        self.num_layers = args.gnn_depth
        
        self.batch_norm = nn.BatchNorm1d(5)
        
        self.embed_category = nn.Embedding(101, self.emb_dim)
        self.embed_country = nn.Embedding(92, self.emb_dim)
        self.embed_sl = nn.Embedding(6, self.emb_dim)
        if self.with_rules:
            self.embed_rules = nn.Embedding(9, self.emb_dim)
        
        self.embed_url = nn.Embedding(128, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(self.emb_dim*2, self.emb_dim)
        
        self.attn_linear = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1)
        self.softmax = nn.Softmax(dim=1)
        
        self.conv_s = nn.ModuleList([SAGEConvN(self.emb_dim, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        self.conv_c = nn.ModuleList([SAGEConvN(self.emb_dim, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        self.conv_co = nn.ModuleList([SAGEConvN(self.emb_dim, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        self.conv_sl = nn.ModuleList([SAGEConvN(self.emb_dim, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        self.conv_ip = nn.ModuleList([SAGEConvN(32, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        if self.with_rules:
            self.conv_r = nn.ModuleList([SAGEConvN(self.emb_dim, self.n_hidden, aggregator_type=self.agg_type)] + 
                                        [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        
        # self.conv = nn.ModuleList([SAGEConvN(self.emb_dim*4+32, self.n_hidden, aggregator_type=self.agg_type)] + 
        #                             [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        
        
        self.fc = nn.Linear(self.n_hidden*2, self.n_hidden)
        self.fc_out = nn.Linear(self.n_hidden, 2)
        
        
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
            
        assert inputs_s.shape[1] == inputs_sm.shape[1]
        assert inputs_ip.shape[1] == 32
        
        vec_category = self.embed_category(inputs_c)
        vec_country = self.embed_country(inputs_co)
        vec_sl = self.embed_sl(inputs_sl)
        vec_ip = inputs_ip
        if self.with_rules:
            vec_rules = self.embed_rules(inputs_r.long())
        
        url_input = inputs_s
        url_mask = inputs_sm
        input_lengths = url_mask.sum(dim=1)
        vec_url_input = self.embed_url(url_input)
        packed_input = pack_padded_sequence(vec_url_input, input_lengths.tolist(), batch_first=True, enforce_sorted=False)
        _, (ht, ct) = self.lstm(packed_input)
        vec_url_output = torch.concat([ht[0], ht[-1]], dim=1)
        _vec_url = vec_url_output[packed_input.unsorted_indices]
        _vec_url = self.fc_lstm(_vec_url)
        _vec_url = self.relu(_vec_url)
        _vec_url = self.dropout(_vec_url)
        vec_url = _vec_url
        
        h1 = vec_url
        h2 = vec_category
        h3 = vec_country
        h4 = vec_sl
        h5 = vec_ip
        if self.with_rules:
            h6 = vec_rules
    
        for i in range(self.num_layers):
            h1 = self.conv_s[i](blocks[i], h1)
            h1 = self.relu(h1)
            h1 = self.dropout(h1)
            
        for i in range(self.num_layers):
            h2 = self.conv_c[i](blocks[i], h2)
            h2 = self.relu(h2)
            h2 = self.dropout(h2)
        
        for i in range(self.num_layers):
            h3 = self.conv_co[i](blocks[i], h3)
            h3 = self.relu(h3)
            h3 = self.dropout(h3)
            
        for i in range(self.num_layers):
            h4 = self.conv_sl[i](blocks[i], h4)
            h4 = self.relu(h4)
            h4 = self.dropout(h4)
            
        for i in range(self.num_layers):
            h5 = self.conv_ip[i](blocks[i], h5)
            h5 = self.relu(h5)
            h5 = self.dropout(h5)
            
        if self.with_rules:
            for i in range(self.num_layers):
                h6 = self.conv_r[i](blocks[i], h6)
                h6 = self.relu(h6)
                h6 = self.dropout(h6)
        
        if self.with_rules:
            h_stack = torch.stack([h1, h2, h3, h4, h5, h6], dim=1)
        else:
             h_stack = torch.stack([h1, h2, h3, h4, h5], dim=1)
        attn = self.softmax(self.attn(self.attn_linear(h_stack)))
        h = torch.sum((self.attn_linear(h_stack) * attn), dim=1)
        
        # h = torch.cat([h1, h2, h3, h4, h5], dim=1)
        # for i in range(self.num_layers):
        #     h = self.conv[i](blocks[i], h)
        #     h = self.relu(h)
        #     h = self.dropout(h)
        # attn = self.softmax(self.attn(self.attn_linear(h)))
        # h = torch.sum((self.attn_linear(h) * attn), dim=1)
        
        with edge_sub.local_scope():
            edge_sub.ndata['h'] = h
            edge_sub.ndata['attn'] = attn
            edge_sub.apply_edges(self.apply_edges, etype='sim')
            return edge_sub.edata['score'][('site', 'sim', 'site')], edge_sub.edata['attn'][('site', 'sim', 'site')]