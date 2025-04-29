import torch
import dgl
import sys
from sklearn.model_selection import train_test_split

class PassREfinder(object):
    def __init__(self, feature_dict, reuse_rate_dict, args, device):
        super(PassREfinder, self).__init__()
        self.args = args
        self.device = device
        self.setting = args.setting
        
        self.g = self.construct_graph(feature_dict, reuse_rate_dict)
         
        assert(args.setting in ['transductive', 'inductive'])
        print('[*] Learning setting:', args.setting)
        
        if args.setting == 'transductive':
            self.edge_split = self.transductive_split()
        else:
            self.g_split, self.node_split = self.inductive_split()
        
    def construct_graph(self, feature_dict, reuse_rate_dict):
        print('[*] Constructing graph...')
        
        source_to_id = {s: i for i, s in enumerate(feature_dict.keys())}
        id_to_source = {v: k for k, v in source_to_id.items()}

        id_to_feature = {source_to_id[k]: {'numerical': v['numerical'], 'categorical': v['categorical'], 'url_encoding': v['url_encoding'], 'ip': v['ip'], 'text': v['text']} for k, v in feature_dict.items()}

        score_dict = {}
        label_list = []
        src_list = []
        dst_list = []

        for s1, id1 in source_to_id.items():
            s2_list = reuse_rate_dict[s1]

            for s2, score in s2_list.items():
                id2 = source_to_id.get(s2)
                score = 0 if score < self.args.reuse_th else 1

                if id2:
                    pair = tuple(sorted((id1, id2)))

                    if pair in score_dict:
                        continue

                    score_dict[pair] = score

                    src_list.append(id1)
                    dst_list.append(id2)
                    label_list.append(score)

        data_dict = {
            ('website', 'user', 'website'): (src_list, dst_list),
            ('website', 'reuse', 'website'): (src_list, dst_list),
        }           

        g = dgl.to_bidirected(dgl.heterograph(data_dict))

        g.ndata['numerical'] = torch.tensor([v['numerical'] for v in id_to_feature.values()])
        g.ndata['categorical'] = torch.tensor([v['categorical'] for v in id_to_feature.values()])
        g.ndata['url_encoding'] = torch.tensor([v['url_encoding'] for v in id_to_feature.values()])
        g.ndata['text'] = torch.tensor([v['text'] for v in id_to_feature.values()])
        g.ndata['ip'] = torch.tensor([v['ip'] for v in id_to_feature.values()], dtype=torch.float)

        g.edata['mask'] = {('website', 'reuse', 'website'): torch.zeros(len(src_list)*2, dtype=torch.float)}
        g.edata['mask'][('website', 'reuse', 'website')][g.edge_ids(src_list, dst_list, etype='reuse')] = torch.tensor(label_list, dtype=torch.float)
        g.edata['mask'][('website', 'reuse', 'website')][g.edge_ids(dst_list, src_list, etype='reuse')] = torch.tensor(label_list, dtype=torch.float)

        g.edata['label'] = {('website', 'reuse', 'website'): torch.zeros(len(src_list)*2, dtype=torch.long)}
        g.edata['label'][('website', 'reuse', 'website')][g.edge_ids(src_list, dst_list, etype='reuse')] = torch.tensor(label_list)
        g.edata['label'][('website', 'reuse', 'website')][g.edge_ids(dst_list, src_list, etype='reuse')] = torch.tensor(label_list)
        
        sys.stdout.write("\033[F")
        print('[+] Constructing graph... Done')

        return g

    def inductive_split(self):
        print('[*] Splitting graph...')
        
        valid_portion, test_portion = self.args.valid, self.args.test

        p1 = valid_portion + test_portion
        p2 = test_portion / (valid_portion + test_portion)

        train_node, valid_node = train_test_split(self.g.nodes(), test_size=p1, shuffle=True, random_state=self.args.random_seed)
        valid_node, test_node = train_test_split(valid_node, test_size=p2, shuffle=True, random_state=self.args.random_seed)

        train_g = self.g.subgraph(train_node)
        valid_g = self.g.subgraph(torch.cat([train_node, valid_node], 0))
        test_g = self.g.subgraph(torch.cat([train_node, test_node], 0))
        
        sys.stdout.write("\033[F")
        print('[+] Splitting graph... Done')

        return {'train': train_g, 'valid': valid_g, 'test': test_g}, {'train': train_node, 'valid': valid_node, 'test': test_node}

    def transductive_split(self):
        print('[*] Splitting graph...')
        
        valid_portion, test_portion = self.args.valid, self.args.test

        p1 = valid_portion + test_portion
        p2 = test_portion / (valid_portion + test_portion)

        train_edge, valid_edge = train_test_split(self.g.edges(etype='reuse', form='eid'), test_size=p1, shuffle=True, random_state=self.args.random_seed)
        valid_edge, test_edge = train_test_split(valid_edge, test_size=p2, shuffle=True, random_state=self.args.random_seed)
        
        sys.stdout.write("\033[F")
        print('[+] Splitting graph... Done')
        return {'train': train_edge, 'valid': valid_edge, 'test': test_edge}
    
    def get_reverse_eid(self, g_type):
        if self.setting == 'transductive':
            g = self.g
        else:
            g = self.g_split[g_type]
        return {('website', 'reuse', 'website'): g.edge_ids(g.edges(etype='reuse')[1], g.edges(etype='reuse')[0], etype='reuse')}

    def get_target_eid(self, g_type):
        if self.setting == 'transductive':
            return self.edge_split[g_type]
        else:
            g = self.g_split[g_type]
            node = self.node_split[g_type]

            ori_id_to_id = dict(zip(g.ndata['_ID'].tolist(), g.nodes().tolist()))
            node_sub = [ori_id_to_id[n.item()] for n in node]

            return torch.unique(torch.cat([g.in_edges(node_sub, etype='reuse', form='eid'), g.out_edges(node_sub, etype='reuse', form='eid')]))

    def get_data_loader(self, g_type):
        if self.setting == 'transductive':
            g = self.g
        else:
            g = self.g_split[g_type]
        reverse_eid = self.get_reverse_eid(g_type)
        target_eid = self.get_target_eid(g_type)

        sampler = dgl.dataloading.as_edge_prediction_sampler(dgl.dataloading.MultiLayerFullNeighborSampler(self.args.gnn_depth), exclude='reverse_id', reverse_eids=reverse_eid)

        data_loader = dgl.dataloading.DataLoader(
                g,
                {'reuse': target_eid},
                sampler,
                batch_size=self.args.batch_size,
                device=self.device,
                shuffle=False,
                drop_last=False,
                num_workers=0)

        return data_loader

    def pop_node_feature(self, g_type):
        if self.setting == 'transductive':
            g = self.g
        else:
            g = self.g_split[g_type]
        
        nfeat_n = g.ndata.pop('numerical')
        nfeat_c = g.ndata.pop('categorical')
        nfeat_e = g.ndata.pop('url_encoding')
        nfeat_ip = g.ndata.pop('ip')
        nfeat_text = g.ndata.pop('text')

        return nfeat_n, nfeat_c, nfeat_e, nfeat_ip, nfeat_text

    def print_graph(self):
        if self.setting == 'transductive':
            g = self.g
            train_edge = self.edge_split['train']
            valid_edge = self.edge_split['valid']
            test_edge = self.edge_split['test']

            def get_node_num(g, edges):
                node_set = set()
                for n1, n2 in zip(g.find_edges(edges, etype='reuse')[0].tolist(), g.find_edges(edges, etype='reuse')[1].tolist()):
                    node_set.add(n1)
                    node_set.add(n2)

                return len(node_set)

            print(f"""----------Data statistics----------
#[Train] nodes: {get_node_num(self.g, train_edge)}, edges: {len(train_edge)}
#[Valid] nodes: {get_node_num(self.g, valid_edge)}, edges: {len(valid_edge)}
#[Test]  nodes: {get_node_num(self.g, test_edge)}, edges: {len(test_edge)}
-----------------------------------\n""")
        else:
            train_g = self.g_split['train']
            valid_g = self.g_split['valid']
            test_g = self.g_split['test']
            train_eid = self.get_target_eid('train')
            valid_eid = self.get_target_eid('valid')
            test_eid = self.get_target_eid('test')

            print(f"""\n[*] Data statistics
[*] Train | nodes: {train_g.number_of_nodes()}, edges: {train_g.number_of_edges(etype='reuse')}, target edges: {len(train_eid)}
[*] Valid | nodes: {valid_g.number_of_nodes()}, edges: {valid_g.number_of_edges(etype='reuse')}, target edges: {len(valid_eid)}
[*] Test  | nodes: {test_g.number_of_nodes()}, edges: {test_g.number_of_edges(etype='reuse')}, target edges: {len(test_eid)}
\n""")