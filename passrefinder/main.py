#!/usr/bin/env python
# coding: utf-8

import json
import os
import shutil
import torch
from torch.optim import Adam
import dgl

from src import *

from argparse import ArgumentParser

parser = ArgumentParser()

import argparse
from pprint import pprint


def init():

    parser = argparse.ArgumentParser(description="Parser for model configuration")

    parser.add_argument('--feature_file', type=str, default="./data/feature_dict.json", help='feature json file')
    parser.add_argument('--reuse_rate_file', type=str, default="./data/reuse_rate_dict.json", help='reuse rate json file')
    parser.add_argument('--setting', type=str, required=True, help='graph learning setting: inductive/transductive')
    parser.add_argument('--model_path', type=str, required=True, default="./model/", help='model grand path')
    parser.add_argument('--model_name', type=str, default="model.pth", help='model folder name')
    parser.add_argument('--agg_type', type=str, default="attn", help='aggregation type')
    parser.add_argument('--valid', type=float, default=0.2, help='split ratio of validation set')
    parser.add_argument('--test', type=float, default=0.2, help='split ratio of test set')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed for initialization')
    parser.add_argument('--relu', type=float, default=0.2, help='ReLU threshold')
    parser.add_argument('--reuse_th', type=float, default=0.5, help='threshold for reuse')
    parser.add_argument('--batch_size', type=int, default=65536, help='batch size for training')
    parser.add_argument('--embed_size', type=int, default=256, help='size of the embedding layer')
    parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden layer')
    parser.add_argument('--gnn_depth', type=int, default=2, help='depth of the GNN')
    parser.add_argument('--max_lr', type=float, default=0.001, help='maximum learning rate')
    parser.add_argument('--warmup', type=float, default=0.1, help='warmup ratio for learning rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='maximum number of epochs')
    parser.add_argument('--early_stop', type=int, default=40, help='early stopping epochs')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'[*] device: {device}')

    print("[*] Arguments")
    pprint(vars(args))
    print("\n")
    return args, device

    
def main_origin(args, device):
    with open(args.feature_file, 'r') as f:
        feature_dict = json.load(f)

    with open(args.reuse_rate_file, 'r') as f:
        reuse_rate_dict = json.load(f)
        
    P = PassREfinder(feature_dict, reuse_rate_dict, args, device)
    
    P.print_graph()
    
    test_loader = P.get_data_loader('test')
    test_nfeat = P.pop_node_feature('test')

    model=GraphSAGE(args).to(device)
    optimizer = Adam(model.parameters(), lr=float(args.max_lr))
    save_checkpoint(args.model_path, model, optimizer, device)
    
    evaluate(model, test_loader, test_nfeat, device)
    
    
def main(args, device):
    with open(args.feature_file, 'r') as f:
        feature_dict = json.load(f)

    with open(args.reuse_rate_file, 'r') as f:
        reuse_rate_dict = json.load(f)
        
    path = os.path.join(args.model_path, args.model_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
        
    P = PassREfinder(feature_dict, reuse_rate_dict, args, device=device)
    
    P.print_graph()
    
    train_loader = P.get_data_loader('train')
    valid_loader = P.get_data_loader('valid')
    test_loader = P.get_data_loader('test')
    
    train_nfeat = P.pop_node_feature('train')
    valid_nfeat = P.pop_node_feature('valid')
    test_nfeat = P.pop_node_feature('test')

    model = GraphSAGE(args).to(device)
    optimizer = Adam(model.parameters(), lr=float(args.max_lr))
    loss_fn = torch.nn.BCELoss()
    
    train_loss_list = []
    valid_loss_list = []
    f1_list = []
    result_list = []
    not_improved = 0
    
    
    model.train()
    for epoch in range(args.max_epoch):
        total_loss = 0.0
        for input_nodes, edge_sub, blocks in train_loader:
            optimizer.zero_grad()
            batch_inputs, batch_labels = load_subtensor(*train_nfeat, edge_sub, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            output, attn = model(edge_sub, blocks, *batch_inputs)
            probs = torch.sigmoid(output)  # [batch_size, 2]
        
            batch_labels = torch.nn.functional.one_hot(batch_labels, num_classes=2).float()
            loss = loss_fn(probs.float(), batch_labels.float())
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        train_loss_list.append(train_loss)
        
        total_loss = 0.0
        for input_nodes, edge_sub, blocks in valid_loader:
            optimizer.zero_grad()
            batch_inputs, batch_labels = load_subtensor(*valid_nfeat, edge_sub, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            output, attn = model(edge_sub, blocks, *batch_inputs)
            probs = torch.sigmoid(output)  # [batch_size, 2]
        
            batch_labels = torch.nn.functional.one_hot(batch_labels, num_classes=2).float()
            loss = loss_fn(probs.float(), batch_labels.float())
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()    
        valid_loss = total_loss / len(valid_loader)
        valid_loss_list.append(valid_loss)
        
        valid_result = evaluate(model, valid_loader, valid_nfeat, device)
        result_list.append(valid_result)
        f1_score = valid_result['macro avg']['f1-score']
        f1_list.append(f1_score)

        print(f"[*] Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | f1-score: {f1_score:.4f} | Best f1-score: {max(f1_list):.4f}")
        
        path = os.path.join(args.model_path, args.model_name, f"model_{epoch}.pth")
        save_checkpoint(path, model, optimizer, valid_result)
        if f1_score >= max(f1_list):
            path = os.path.join(args.model_path, args.model_name, "model_best.pth")
            save_checkpoint(path, model, optimizer, valid_result)
            
        if valid_loss <= min(valid_loss_list):
            not_improved = 0
        else:
            not_improved += 1
        if not_improved >= args.early_stop:
            print(f"[*] Early stopping at epoch {epoch}")
            break
            
    print("[+] Training finished")
    path = os.path.join(args.model_path, args.model_name, "model_best.pth")
    valid_socre = load_checkpoint(path, model, optimizer, device=device)
    test_result = evaluate(model, test_loader, test_nfeat, device)
    
    print("\n\n[*] Model Rresult")
    print(f"[*] Valid precision: {valid_socre['macro avg']['precision']:.4f}")
    print(f"[*] Valid recall: {valid_socre['macro avg']['recall']:.4f}")
    print(f"[*] Valid f1-score: {valid_socre['macro avg']['f1-score']:.4f}")
    print(f"\n\n[*] Test precision: {test_result['macro avg']['precision']:.4f}")
    print(f"[*] Test recall: {test_result['macro avg']['recall']:.4f}")
    print(f"[*] Test f1-score: {test_result['macro avg']['f1-score']:.4f}")
    
    path = os.path.join(args.model_path, args.model_name, "result.json")
    with open(path, 'w') as f:
        json.dump({'train_loss': train_loss_list, 'valid_loss': valid_loss_list, 'f1': f1_list, 'result': result_list}, f)
    
    

if __name__ == '__main__':
    args, device = init()
    main(args, device)