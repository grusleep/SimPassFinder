import numpy as np
import sys
import torch
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def load_subtensor(nfeat_s, nfeat_sm, nfeat_c, nfeat_co, nfeat_sl, nfeat_ip, edge_sub, input_nodes, device='cpu'):
    nfeat_s = nfeat_s.to(device)
    nfeat_sm = nfeat_sm.to(device)
    nfeat_c = nfeat_c.to(device)
    nfeat_co = nfeat_co.to(device)
    nfeat_sl = nfeat_sl.to(device)
    nfeat_ip = nfeat_ip.to(device)
    input_nodes = input_nodes.to(device)
    
    
    batch_inputs_s = nfeat_s[input_nodes].to(device)
    batch_inputs_sm = nfeat_sm[input_nodes].to(device)
    batch_inputs_c = nfeat_c[input_nodes].to(device)
    batch_inputs_co = nfeat_co[input_nodes].to(device)
    batch_inputs_sl = nfeat_sl[input_nodes].to(device)
    batch_inputs_ip = nfeat_ip[input_nodes].to(device)
    edge_sub = edge_sub.to(device)
    batch_labels = edge_sub.edata['label'][('site', 'sim', 'site')]
    
    return (batch_inputs_s, batch_inputs_sm, batch_inputs_c, batch_inputs_co, batch_inputs_sl, batch_inputs_ip), batch_labels

def evaluate(model, data_loader, nfeat, device='cpu'):
    y_true = []
    y_pred = []
    nodes = []
    
    model.eval()
    with torch.no_grad():
        pair_to_score = defaultdict(dict)
        
        for input_nodes, edge_sub, blocks in data_loader:
            batch_inputs, batch_labels = load_subtensor(*nfeat, edge_sub, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            
            batch_pred, attn = model(edge_sub, blocks, *batch_inputs)

            preds = batch_pred.argmax(dim=1)
            # print(batch_inputs[0])
        
            y_true.extend(batch_labels.detach().cpu().long().tolist())
            y_pred.extend(preds.detach().cpu().long().tolist())
            
        
    result = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    model.train()
    return result, nodes, y_true, y_pred