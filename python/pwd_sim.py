import os
import json
import argparse
import itertools
import ijson

from src import *
from difflib import SequenceMatcher
from tqdm import tqdm

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
        
        
    #def load_node(self):
    #    self.logger.print(f"[*] Loading node data")
    #    node_file = os.path.join(self.dataset_path, "graph", "nodes.json")
    #    with open(node_file, "r") as f:
    #        self.nodes = json.load(f)
    #    self.logger.print(f"[+] Done loading nodes")
    #    self.logger.print(f"[+] Number of nodes: {len(self.nodes)}\n")
    
    # Modify function. Load user data from JSON file
    #def load_users(self):
    #    self.logger.print(f"[*] Loading user data")
    #    user_file = os.path.join(self.dataset_path, "users", "users_test.json")
    #    with open(user_file, "r") as f:
    #        self.users = json.load(f)
    #    self.logger.print(f"[+] Done loading users")
    #    self.logger.print(f"[+] Number of users: {len(self.users)}\n")

    # Lazy loading user data from JSON file
    def load_users_lazy(self, path):
        self.logger.print(f"[*] Lazy loading users from {path}\n")
        invalid_users = []
        error_path = os.path.join(self.dataset_path, "users", "invalid_users.txt")
        count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                parser = ijson.kvitems(f, '')  # 최상위 key-value 쌍 파싱
                for email, data in parser:
                    try:
                        if not isinstance(data, list):
                            raise ValueError("Invalid data type")
                        #self.logger.print(f"[*] Processing user: {email}")
                        yield email, data
                    except Exception as e:
                        invalid_users.append(email)
                        self.logger.print(f"[!] Invalid user data for {email}: {e}")
                    count += 1
        except Exception as e:
            self.logger.print(f"[!] JSON 파싱 중단: {e} (총 {count}명까지 처리됨)")
        if invalid_users:
            self.logger.print(f"[!] Invalid user count: {len(invalid_users)}")
            with open(error_path, "w", encoding="utf-8") as f:
                for email in invalid_users:
                    f.write(email + "\n")


    #def graph_to_users(self):
    #    self.logger.print(f"[*] Converting graph to sites")
    #    users_num = 0
    #    
    #    for i, node in enumerate(self.nodes):
    #        site = node['site']
    #        with open(os.path.join(self.dataset_path, "sites", f"{site}.json")) as f:
    #            data = json.load(f)
    #        users = data.keys()
    #        for user in users:
    #            if user not in self.users:
    #                users_num += 1
    #                self.users[user] = {"num": users_num, "data": []}
    #            self.users[user]["data"].append({
    #                "site": site,
    #                "password": data[user]
    #            })
    #            
    #        self.logger.print(f"[*] Processing site: {i:5} / {len(self.nodes):5}")
    #        if i!=0 and i%1000 == 0:
    #            with open(os.path.join(self.dataset_path, "users", f"users_{i}.json"), "w") as f:
    #                json.dump(self.users, f, indent=4)
    #            self.users = {}
    #            self.logger.print(f"[+] Saving users to file: users_{i}.json")
    #    
    #    with open(os.path.join(self.dataset_path, "users", f"users_{len(self.nodes)}.json"), "w") as f:
    #        json.dump(self.users, f, indent=4)
    #    self.logger.print(f"[+] Total users: {len(self.users)}")
        
        
    #def union_users_file(self, file1, file2, output_file):
    #    self.logger.print(f"[*] Union users from {file1} and {file2}")
    #    with open(os.path.join(self.dataset_path, "users", file1), "r") as f1:
    #        data1 = json.load(f1)
    #    self.logger.print(f"[+] Loaded {len(data1)} users from {file1}")
    #    with open(os.path.join(self.dataset_path, "users", file2), "r") as f2:
    #        data2 = json.load(f2)
    #    self.logger.print(f"[+] Loaded {len(data2)} users from {file2}")
    #    exit()
    #    user2 = set(data2.keys())
    #    len_common = len(user2)
    #    for i, user in enumerate(user2):
    #        if user not in data1:
    #            data1[user] = data2[user]
    #        else:
    #            data1[user] = data1[user] + data2[user]
    #        if i != 0 and i % 100000 == 0:
    #            self.logger.print(f"[*] Processing user: {i:9} / {len_common:9}")
    #    with open(os.path.join(self.dataset_path, "users", output_file), "w") as f:
    #        json.dump(data1, f, indent=4)
    #    self.logger.print(f"[+] Union done, saved to {output_file}")


    def save_sites_rules(self, sites_rules):
        self.logger.print(f"[*] Saving site list by rule")
        
        if not os.path.exists(os.path.join(self.dataset_path, "users")):
            os.makedirs(os.path.join(self.dataset_path, "users"))

        site_rule = {
            site: {str(rule): count for rule, count in sorted(rules.items())}
            for site, rules in sorted(sites_rules.items())
        }

        output_path = os.path.join(self.dataset_path, "rules", "sites_rules.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(site_rule, f, indent=4, ensure_ascii=False)

        self.logger.print(f"[+] Saved site list by rule to {output_path}")

        
    def find_rule(self):
        user_file = os.path.join(self.dataset_path, "users", "users_all.json")
        user_stream = self.load_users_lazy(user_file)

        self.logger.print(f"[*] Finding password rules")
        num_single_site_users = 0
        sites_rules = dict()
        
        for idx, (user, user_data) in enumerate(tqdm(user_stream, desc="Processing users", dynamic_ncols=True)):
            if len(user_data) < 2:
                num_single_site_users += 1
            else:
                for i in range(len(user_data)):
                    for j in range(i+1, len(user_data)):
                        pwd1, site1 = user_data[i]['password'], user_data[i]['site']
                        pwd2, site2 = user_data[j]['password'], user_data[j]['site']
                        rules = self.pwd_transformation_rule(pwd1, pwd2)
                        for rule in rules:
                            for site in [site1, site2]:
                                if site not in sites_rules:
                                    sites_rules[site] = {}
                                key = f"rule {rule}"
                                if key not in sites_rules[site]:
                                    sites_rules[site][key] = 0
                                sites_rules[site][key] += 1
              
            #if idx % 10000 == 0 and idx > 0:
            #    self.save_sites_rules(sites_rules)
            #    sites_rules = dict()  
                
            #self.logger.print(f"[*] Processing user: {idx:5} / {len(self.users):5}")
        #self.logger.print(f"[+] Users with rules: {users_rules}")
        self.logger.print(f"[+] Done processing users\n")
        self.save_sites_rules(sites_rules)
        self.logger.print(f"[+] Total users with single site: {num_single_site_users}")
        
        
    def pwd_transformation_rule(self, pwd1, pwd2):
        rules = []
        if pwd1 == pwd2: 
            rules.append(1) # rule 1: identical
        else:
            if pwd1 in pwd2 or pwd2 in pwd1:
                rules.append(2) # rule 2: substring
            if pwd1.lower() == pwd2.lower():
                rules.append(3) # rule 3: capitalization
            if not self.get_leet_variations(pwd1).isdisjoint(self.get_leet_variations(pwd2)):
                rules.append(4) # rule 4: leet
            if self.reverse_password(pwd1, pwd2):
                rules.append(5) # rule 5: reversal
            if self.sequential_key(pwd1) and self.sequential_key(pwd2):
                rules.append(6) # rule 6: sequential keys
            if self.common_substring(pwd1, pwd2):
                rules.append(7) # rule 7: common substring
            if self.combine_rules(pwd1, pwd2):
                rules = []
                rules.append(8) # rule 8: combine rules
            if not rules:
                rules.append(0) # no rules matched
        return rules
    
    
    def get_leet_variations(self, password, max_comb=1000):
        variations = []
        for char in password:
            leet_chars = self.leet_map.get(char, [])
            leet_chars = leet_chars + [char]
            variations.append(leet_chars)
        # 조합 수가 너무 많으면 일부만 생성
        total_comb = 1
        for v in variations:
            total_comb *= len(v)
        if total_comb > max_comb:
            return set([password])
        all_combinations = [''.join(p) for p in itertools.product(*variations)]
        return set(all_combinations)
    
    
    def reverse_password(self, pwd1, pwd2):
        reversed_pwd = pwd1[::-1]

        if pwd2 == reversed_pwd:
            return True
        
        return False
    
    def sequential_key(self, pwd):
        if len(pwd) < 3:
            return False

        sequences = [
            "`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./",
            "~!@#$%^&*()_+", "QWERTYUIOP{}|", "ASDFGHJKL:\"", "ZXCVBNM<>?",
            "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ]

        for seq in sequences:
            for i in range(len(seq) - 2):
                sub = seq[i:i+3]
                if sub in pwd or sub[::-1] in pwd:
                    return True

        return False
    
    
    def common_substring(self, pwd1, pwd2):

        matcher = SequenceMatcher(None, pwd1, pwd2)
        matches = matcher.get_matching_blocks()

        sizes = [match.size for match in matches if match.size > 0]
        if not sizes:
            return False

        max_len = max(sizes)
        total_common = sum(size for size in sizes if size > 0)
        min_pwd_len = min(len(pwd1), len(pwd2))

        return max_len >= 3 and (total_common / min_pwd_len > 0.5)

    
    def apply_rule_variants(self, pwd):
        variants = set()
        variants.add(pwd)

        # rule 3: capitalization
        variants.add(pwd.lower())

        # rule 4: leet
        variants.update(self.get_leet_variations(pwd))

        # rule 5: reversal
        variants.add(pwd[::-1])

        # rule 6: sequential keys는 구조적으로 변형을 만들 수는 없으므로 비교만
        # 여기서는 따로 변형 X

        return variants

    
    
    def combine_rules(self, pwd1, pwd2):
        variants1 = self.apply_rule_variants(pwd1)
        variants2 = self.apply_rule_variants(pwd2)

        for v1 in variants1:
            for v2 in variants2:
                if v1 == v2:
                    return True 
                if v1 in v2 or v2 in v1:
                    return True  # rule 2: substring
                if self.common_substring(v1, v2):
                    return True  # rule 7: common substring

        return False



def init():
    parser = argparse.ArgumentParser(description="Parser for data processing configuration")
    parser.add_argument('--run_type', type=str, default="pwd_sim", help='run type: train/test/data_processing')
    return parser.parse_args()


if __name__ == "__main__":
    args = init()
    logger = Logger(args)
    processor = PasswordSimilarity(args, logger)
    #processor.load_users()
    #processor.load_node()
    #processor.union_users_file("users_1_8.json", "users_9_10_11_12.json", "users_all.json")
    processor.find_rule()