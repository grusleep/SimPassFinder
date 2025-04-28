import os
import dgl
import json
import jellyfish
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from .utils import *



class CustomDataset():
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.edge_thv = args.edge_thv
        self.graph_type = args.setting
        self.valid = args.valid
        self.test = args.test
        self.load_category()
        self.load_country()
        self.load_node()
        # self.encoding_node()
        
        
    def load_meta_data(self):
        print(f"[*] Loading meta data")
        with open(os.path.join(self.dataset_path, "meta_data","meta_data.json"), "r") as f:
            self.meta_data = json.load(f)
        print(f"[+] Done loading meta data")
        print(f"[+] Number of sites: {len(self.meta_data)}\n")
    
    
    def load_edge(self):
        print(f"[*] Loading edges")
        with open(os.path.join(self.dataset_path, "graph", "edges.json"), "r") as f:
            self.edges = json.load(f)
        print(f"[+] Done loading edges")
        print(f"[+] Number of edges: {len(self.edges)}\n")


    def load_node(self):
        print(f"[*] Loading nodes")
        with open(os.path.join(self.dataset_path, "graph", f"nodes.json"), "r") as f:
            self.nodes = json.load(f)
        print(f"[+] Done loading nodes")
        print(f"[+] Number of nodes: {len(self.nodes)}\n")
        
    
    def load_country(self):
        print(f"[*] Loading countries")
        with open(os.path.join(self.dataset_path, "graph", "countries.json"), "r") as f:
            self.countries = json.load(f)
        print(f"[+] Done loading countries")
        print(f"[+] Number of countries: {len(self.countries)}\n")
            
    
    def load_category(self):
        print(f"[*] Loading categories")
        with open(os.path.join(self.dataset_path, "graph", "categories.json"), "r") as f:
            self.categories = json.load(f)
        print(f"[+] Done loading categories")
        print(f"[+] Number of categories: {len(self.categories)}\n")
    
    
    def set_node(self, save=True):
        self.nodes = []
        print(f"[*] Start Setting nodes")
        
        for site in tqdm(list(self.meta_data.keys()), desc="[*] Setting nodes"):
            node = {}
            node["site"] = site
            node["category"] = self.meta_data[site]["category"]
            node["country"] = self.meta_data[site]["country"]
            node["security_level"] = self.meta_data[site]["security_level"]
            node["ip"] = self.meta_data[site]["ip"]
            self.nodes.append(node)
        print(f"[+] Done setting nodes")
        print(f"[+] Number of nodes: {len(self.nodes)}")
        
        if save:
            node_file = os.path.join(self.dataset_path, "graph", f"nodes.json")
            with open(node_file, "w", encoding="utf-8") as f:
                json.dump(self.nodes, f, indent=4, ensure_ascii=False)
            print(f"[+] Saved nodes to {node_file}")
        
        
    def set_edge(self, save=True):
        self.edges = []
        print(f"[*] Start Setting edges")
        
        for i in tqdm(range(len(self.nodes)), desc="[*] Setting edges"):
            for j in range(i + 1, len(self.nodes)):
                i_site = self.nodes[i]["site"]
                j_site = self.nodes[j]["site"]
                with open(os.path.join(self.dataset_path, "sites", f"{i_site}.json")) as f:
                    i_data = json.load(f)
                with open(os.path.join(self.dataset_path, "sites", f"{j_site}.json")) as f:
                    j_data = json.load(f)
                i_data_user_id = set(i_data.keys())
                j_data_user_id = set(j_data.keys())
                intersection = i_data_user_id & j_data_user_id
                if len(intersection) == 0:
                    continue
                pwd_sim = []
                for user_id in intersection:
                    pwd_sim.append(jellyfish.jaro_similarity(i_data[user_id], j_data[user_id]))
                weight = sum(pwd_sim) / len(pwd_sim)
    
                edge = {}
                edge["node_1"] = i
                edge["node_2"] = j
                edge["weight"] = weight
                self.edges.append(edge)
        
        if save:
            edge_file = os.path.join(self.dataset_path, "graph", f"edges.json")
            with open(edge_file, "w", encoding="utf-8") as f:
                json.dump(self.edges, f, indent=4, ensure_ascii=False)
            print(f"[+] Saved edges to {edge_file}\n")
            
            
    def __encoding_node(self, node):
        encode_node = {}
        encode_node["site"] = [ord(c) for c in node["site"]]
        encode_node["category"] = self.categories.index(node["category"])
        encode_node["country"] = self.countries.index(node["country"])
        encode_node["security_level"] = node["security_level"]
        bit_strs = [f"{int(octet):08b}" for octet in node["ip"].split(".")]
        encode_node["ip"] = [int(bit)
                            for byte in bit_strs
                            for bit in byte]

        return encode_node
    
    
    def encoding_node(self):
        print(f"[*] Start encoding node")
        for node in tqdm(self.nodes, desc="[*] Encoding nodes"):
            encode_node = self.__encoding_node(node)
            node["site"] = encode_node["site"]
            node["category"] = encode_node["category"]
            node["country"] = encode_node["country"]
            node["security_level"] = encode_node["security_level"]
            node["ip"] = encode_node["ip"]
        print(f"[+] Done encoding nodes\n")
    
    def build_graph(self):
        print(f"[*] Start building graph")
        
        node_1_list = [edge["node_1"] for edge in self.edges]
        node_2_list = [edge["node_2"] for edge in self.edges]
        weight_list = [edge["weight"] for edge in self.edges]
        label_list = [1 if edge["weight"] > self.edge_thv else 0 for edge in self.edges]
        
        data_dict = {
            ("site", "user", "site"): (node_1_list, node_2_list),
            ("site", "sim", "site"): (node_1_list, node_2_list),
        }
        self.graph = dgl.to_bidirected(dgl.heterograph(data_dict))
        
        site_list = [torch.tensor(node["site"], dtype=torch.long) for node in self.nodes]
        padded_site = pad_sequence(site_list, batch_first=True, padding_value=0)
        mask = padded_site.ne(0)
        
        self.graph.ndata["site"] = padded_site
        self.graph.ndata["site_mask"] = mask
        self.graph.ndata["category"] = torch.tensor([node["category"] for node in self.nodes])
        self.graph.ndata["country"] = torch.tensor([node["country"] for node in self.nodes])
        self.graph.ndata["security_level"] = torch.tensor([node["security_level"] for node in self.nodes])
        self.graph.ndata["ip"] = torch.tensor([node["ip"] for node in self.nodes])
        
        # Edge features for the "sim" relation type
        sim_etype = ("site", "sim", "site")
        num_sim_edges = self.graph.num_edges(etype=sim_etype)

        # Initialize weight and label tensors matching total edge count
        self.graph.edata["weight"] = {sim_etype: torch.zeros(num_sim_edges, dtype=torch.float)}
        self.graph.edata["label"]  = {sim_etype: torch.zeros(num_sim_edges, dtype=torch.float)}

        # Assign forward and reverse edge attributes
        eid_fwd = self.graph.edge_ids(node_1_list, node_2_list, etype=sim_etype)
        eid_rev = self.graph.edge_ids(node_2_list, node_1_list, etype=sim_etype)

        self.graph.edata["weight"][sim_etype][eid_fwd] = torch.tensor(weight_list, dtype=torch.float)
        self.graph.edata["weight"][sim_etype][eid_rev] = torch.tensor(weight_list, dtype=torch.float)

        self.graph.edata["label"][sim_etype][eid_fwd] = torch.tensor(label_list, dtype=torch.float)
        self.graph.edata["label"][sim_etype][eid_rev] = torch.tensor(label_list, dtype=torch.float)
        
        print(f"[+] Done building graph\n")
        
        
    def print_graph(self):
        pass
    
    
    def inductive_split(self):
        print('[*] Splitting graph...')
        
        valid_portion, test_portion = self.args.valid, self.args.test

        p1 = valid_portion + test_portion
        p2 = test_portion / (valid_portion + test_portion)

        train_node, valid_node = train_test_split(self.g.nodes(), test_size=p1, shuffle=True, random_state=self.args.random_seed)
        valid_node, test_node = train_test_split(valid_node, test_size=p2, shuffle=True, random_state=self.args.random_seed)

        train_g = self.graph.subgraph(train_node)
        valid_g = self.graph.subgraph(torch.cat([train_node, valid_node], 0))
        test_g = self.graph.subgraph(torch.cat([train_node, test_node], 0))
        
        print('[+] Splitting graph... Done')

        return {'train': train_g, 'valid': valid_g, 'test': test_g}, {'train': train_node, 'valid': valid_node, 'test': test_node}


    def transductive_split(self):
        print('[*] Splitting graph...')
        
        valid_portion, test_portion = self.args.valid, self.args.test

        p1 = valid_portion + test_portion
        p2 = test_portion / (valid_portion + test_portion)

        train_edge, valid_edge = train_test_split(self.g.edges(etype='reuse', form='eid'), test_size=p1, shuffle=True, random_state=self.args.random_seed)
        valid_edge, test_edge = train_test_split(valid_edge, test_size=p2, shuffle=True, random_state=self.args.random_seed)
        
        print('[+] Splitting graph... Done')
        return {'train': train_edge, 'valid': valid_edge, 'test': test_edge}, {'train': train_edge, 'valid': valid_edge, 'test': test_edge}