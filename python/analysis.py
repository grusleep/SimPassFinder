import os
import json
import argparse

from src import *

class Analysis:
    def __init__(self, args, logger):
        self.dataset_path = "../dataset"
        
        self.logger = logger
        self.logger.print(f"[+] Done initializing\n")
        
        
    def load_nodes(self):
        self.logger.print(f"[*] Loading nodes")
        node_file = os.path.join(self.dataset_path, "graph", "nodes.json")
        with open(node_file, 'r', encoding='utf-8') as f:
            self.nodes = json.load(f)
        self.nodes_num = len(self.nodes)
        self.logger.print(f"[+] Done loading nodes")
        self.logger.print(f"[+] Number of nodes: {self.nodes_num}\n")
                
    
    def load_edges(self):
        self.logger.print(f"[*] Loading edges")
        edge_file = os.path.join(self.dataset_path, "graph", "edges.json")
        with open(edge_file, 'r', encoding='utf-8') as f:
            self.edges = json.load(f)
        self.edges_num = len(self.edges)
        self.logger.print(f"[+] Done loading edges")
        self.logger.print(f"[+] Number of edges: {self.edges_num}\n")

            
    # 기존 모델 대비 몇 배의 edge가 추가됐는지
    def edge_count(self):
        self.logger.print(f"[*] Counting original edges")
        origin_edges = len([e for e in self.edges if e["weight"] == 1])
        self.logger.print(f"[+] Original edges: {origin_edges}")
        
        self.logger.print(f"[*] Counting new edges")
        new_edges = len([e for e in self.edges if e["weight"] < 1])
        self.logger.print(f"[+] New edges: {new_edges}")
        self.logger.print(f"[+] Ratio of new edges: {new_edges / origin_edges:.2f}\n")
        
    
    # Count edges per node
    # weight: 0이면 모든 edge, 1이면 기존 edge만
    def edges_per_node(self, weight=0):
        self.logger.print(f"[*] Counting edges per node")
        node_edge = {i: 0 for i in range(self.nodes_num)}
        
        for edge in self.edges:
            if edge["weight"] >= weight:
                node_edge[edge["node_1"]] += 1
                node_edge[edge["node_2"]] += 1

        return sum(list(node_edge.values())) / self.nodes_num


def init():
    parser = argparse.ArgumentParser(description="Analysis script")
    parser.add_argument('--run_type', type=str, default='analysis', help='run type: analysis')
    return parser.parse_args()

if __name__ == "__main__":
    args = init()
    logger = Logger(args)    
    analysis = Analysis(args, logger)
    analysis.load_nodes()
    analysis.load_edges()
    analysis.edge_count()
    
    
    origin_edge_node = analysis.edges_per_node(weight=1)
    logger.print(f"[+] Average edges per node (original): {origin_edge_node:.2f}")
    all_edge_node = analysis.edges_per_node(weight=0)
    logger.print(f"[+] Average edges per node (all): {all_edge_node:.2f}")