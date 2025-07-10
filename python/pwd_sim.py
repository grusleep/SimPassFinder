import os
import json
import argparse
import itertools

from src import *



class PasswordSimilarity:
    def __init__(self, args, logger):
        self.dataset_path = "../dataset"
        self.users = {}
        
        self.logger = logger
        self.logger.print(f"[+] Done initializing")
        
        self.leet_map = {
            '0': ['o'],
            '1': ['i', '!', 'l'],
            '3': ['e'],
            '4': ['a'],
            '5': ['s'],
            '@': ['a'],
            '9': ['6'],
            '$': ['s'],
            'o': ['0'],
            'i': ['1'],
            '!': ['1'],
            'l': ['1'],
            'e': ['3'],
            'a': ['4', '@'],
            's': ['5', '$'],
            '6': ['9']
        }
        
        
    def load_node(self):
        self.logger.print(f"[*] Loading node data")
        node_file = os.path.join(self.dataset_path, "graph", "nodes.json")
        with open(node_file, "r") as f:
            self.nodes = json.load(f)
        self.logger.print(f"[+] Done loading nodes")
        self.logger.print(f"[+] Number of nodes: {len(self.nodes)}\n")
            
    # Modify function. Load user data from JSON file
    def load_users(self):
        self.logger.print(f"[*] Loading user data")
        user_file = os.path.join(self.dataset_path, "users", "users_all.json")
        with open(user_file, "r") as f:
            self.users = json.load(f)
        self.logger.print(f"[+] Done loading users")
        self.logger.print(f"[+] Number of users: {len(self.users)}\n")


    def graph_to_users(self):
        self.logger.print(f"[*] Converting graph to sites")
        users_num = 0
        
        for i, node in enumerate(self.nodes):
            site = node['site']
            with open(os.path.join(self.dataset_path, "sites", f"{site}.json")) as f:
                data = json.load(f)
            users = data.keys()
            for user in users:
                if user not in self.users:
                    users_num += 1
                    self.users[user] = {"num": users_num, "data": []}
                self.users[user]["data"].append({
                    "site": site,
                    "password": data[user]
                })
                
            self.logger.print(f"[*] Processing site: {i:5} / {len(self.nodes):5}")
            if i!=0 and i%1000 == 0:
                with open(os.path.join(self.dataset_path, "users", f"users_{i}.json"), "w") as f:
                    json.dump(self.users, f, indent=4)
                self.users = {}
                self.logger.print(f"[+] Saving users to file: users_{i}.json")
        
        with open(os.path.join(self.dataset_path, "users", f"users_{len(self.nodes)}.json"), "w") as f:
            json.dump(self.users, f, indent=4)
        self.logger.print(f"[+] Total users: {len(self.users)}")
        
        
    def union_users_file(self, file1, file2, output_file):
        self.logger.print(f"[*] Union users from {file1} and {file2}")
        with open(os.path.join(self.dataset_path, "users", file1), "r") as f1:
            data1 = json.load(f1)
        self.logger.print(f"[+] Loaded {len(data1)} users from {file1}")
        with open(os.path.join(self.dataset_path, "users", file2), "r") as f2:
            data2 = json.load(f2)
        self.logger.print(f"[+] Loaded {len(data2)} users from {file2}")
        exit()
        user2 = set(data2.keys())
        len_common = len(user2)
        for i, user in enumerate(user2):
            if user not in data1:
                data1[user] = data2[user]
            else:
                data1[user] = data1[user] + data2[user]
            if i != 0 and i % 100000 == 0:
                self.logger.print(f"[*] Processing user: {i:9} / {len_common:9}")
        with open(os.path.join(self.dataset_path, "users", output_file), "w") as f:
            json.dump(data1, f, indent=4)
        self.logger.print(f"[+] Union done, saved to {output_file}")
        
        
    def find_rule(self):
        self.logger.print(f"[*] Finding password rules")
        num_single_site_users = 0
        
        for i, user in enumerate(self.users.keys()):
            data = self.users[user]
            if len(data) < 2:
                num_single_site_users += 1
            else:
                user_rules = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
                for i in range(len(data)):
                    for j in range(i+1, len(data)):
                        pwd1 = data[i]['password']
                        pwd2 = data[j]['password']
                        rules = self.pwd_transformation_rule(pwd1, pwd2)
                        for rule in rules:
                            user_rules[rule] += 1            
            self.logger.print(f"[*] Processing user: {i:5} / {len(self.users):5}")
        
        
    def pwd_transformation_rule(self, pwd1, pwd2):
        rules = []
        if pwd1 == pwd2:
            rules.append(1)
        else:
            if pwd1 in pwd2 or pwd2 in pwd1:
                rules.append(2)
            if pwd1.lower() == pwd2.lower():
                rules.append(3)
            if not self.get_leet_variations(pwd1).isdisjoint(self.get_leet_variations(pwd2)):
                rules.append(4)
            if self.reverse_password(pwd1, pwd2):
                rules.append(5)
            if self.sequential_key(pwd1) and self.sequential_key(pwd2):
                rules.append(6)
            if self.common_substring(pwd1, pwd2):
                rules.append(7)

        if len(rules) == 0:
            rules.append(0)
        return rules
    
    
    def get_leet_variations(self, password):
        variations = []
        for char in password.lower():
            if char in self.leet_map:
                leet_chars = self.leet_map[char]
                leet_chars.append(char)
                variations.append(leet_chars)
            else:
                variations.append([].append(char))
        all_combinations = [''.join(p) for p in itertools.product(*variations)]
        return set(all_combinations)
    
    
    def reverse_password(self, pwd1, pwd2):        
        pwd1 = pwd1.lower()
        pwd2 = pwd2.lower()
        reversed_pwd = pwd1[::-1]

        if pwd2 == reversed_pwd:
            return True
        
        return False
    
    def sequential_key(self, pwd):
        if len(pwd) < 3:
            return False
        
        pwd = pwd.lower()
        
        keyboard_seq_1 = "`1234567890-="
        keyboard_seq_2 = "qwertyuiop[]\\"
        keyboard_seq_3 = "asdfghjkl;'"
        keyboard_seq_4 = "zxcvbnm,./"
        keyboard_seq_5 = "~!@#$%^&*()_+"
        keyboard_seq_6 = "qwertyuiop{}|"
        keyboard_seq_7 = "asdfghjkl:\""
        keyboard_seq_8 = "zxcvbnm<>?"
        
        keyboard_seqs = [
            keyboard_seq_1, keyboard_seq_2, keyboard_seq_3, keyboard_seq_4,
            keyboard_seq_5, keyboard_seq_6, keyboard_seq_7, keyboard_seq_8
        ]
        
        for seq in keyboard_seqs:
            if pwd in seq or pwd[::-1] in seq:
                return True
        
        def check_seq(s):
            return all(ord(s[i]) + 1 == ord(s[i+1]) for i in range(len(s)-1))
        
        return check_seq(pwd) or check_seq(pwd[::-1])

    
    
    def common_substring(self, pwd1, pwd2):
        len1, len2 = len(pwd1), len(pwd2)
        
        if len1 < 3 or len2 < 3:
            return False
        
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, pwd1, pwd2)
        match = matcher.find_longest_match(0, len1, 0, len2)
        length = match.size

        max_len = max(len1, len2)
        ratio = length / max_len if max_len > 0 else 0

        return length > 2 and ratio > 0.5



def init():
    parser = argparse.ArgumentParser(description="Parser for data processing configuration")
    parser.add_argument('--run_type', type=str, default="pwd_sim", help='run type: train/test/data_processing')
    return parser.parse_args()


if __name__ == "__main__":
    args = init()
    logger = Logger(args)
    processor = PasswordSimilarity(args, logger)
    processor.load_users()
    #processor.load_node()
    #processor.union_users_file("users_1_8.json", "users_9_10_11_12.json", "users_all.json")
    processor.find_rule()