import os
import json
import shutil
import socket
import argparse
import requests
from urllib.parse import urlparse
from pprint import pprint

from src import *



class DataProcessor:
    def __init__(self, args, logger):
        self.dataset_path = "/mnt/d/Cit0day/Cit0day"
        self.meta_data_path = "dataset/meta_data/"
        self.site_path = "dataset/sites"
        self.feature = args.feature
        self.file_list = os.listdir(self.dataset_path)
        self.meta_data = {}
        
        self.logger = logger
        self.logger.print(f"[+] Done initializing")
        
        
    def save_meta_data(self):
        if not os.path.exists(self.meta_data_path):
            os.makedirs(self.meta_data_path)
        
        meta_data_file = os.path.join(self.meta_data_path, "meta_data.json")
        with open(meta_data_file, "w") as f:
            json.dump(self.meta_data, f, indent=4)
        
        self.logger.print(f"[+] Meta data saved to {meta_data_file}")
        
        
    def load_meta_data(self):
        meta_data_file = os.path.join(self.meta_data_path, "meta_data.json")
        if os.path.exists(meta_data_file):
            with open(meta_data_file, "r") as f:
                self.meta_data = json.load(f)
            self.logger.print(f"[+] Meta data loaded from {meta_data_file}")
        else:
            self.logger.print(f"[-] Meta data file not found: {meta_data_file}")
        
        
    def check_ip(self):
        count = 0
        all_count = 0
        file_list_len = len(self.file_list)
        for file in self.file_list:
            site = file.split(" ")[0]
            check = True
            
            try:
                ip_address = socket.gethostbyname(site)
            except:
                check = False
               
            if check: 
                try:
                    url = f"https://ipinfo.io/{ip_address}/json?token=0dd6b0e2ee3821"
                    response = requests.get(url)
                    if response.status_code == 200:
                        ip_data = response.json()
                        country = ip_data["country"]
                except:
                    check = False
            
            if check and self.feature == "all":
                try:
                    url = f"https://{site}"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        html = urlparse(url).netloc
                    else:
                        continue
                except:
                    check = False
            
            if check:
                self.meta_data[site] = {}
                self.meta_data[site]["ip"] = ip_address
                self.meta_data[site]["country"] = country
                if self.feature == "all":
                    self.meta_data[site]["html"] = html
                elif self.feature == "no_html":
                    self.meta_data[site]["html"] = None
                count += 1
            all_count += 1
            self.logger.print(f"[*] Processing {count:5}/{file_list_len} | {all_count:5}/{file_list_len}")
        
        self.logger.print(f"[+] IP data processed. Total sites: {len(self.meta_data)}")
        
        
    def set_category(self):
        for file in self.file_list:
            url = file.split(" ")[0]
            if url not in self.meta_data:
                continue
            if len(file.split("(")) > 2:
                continue
            category = file.split("(")[1].split(")")[0]
            self.meta_data[url]["category"] = category
        self.logger.print(f"[+] Category processed.")
        
        
    def set_sl(self):
        count = 0
        site_num = len(self.file_list)
        for i, file in enumerate(self.file_list):
            url = file.split(" ")[0]
            data_type = file.split(" ")[2]
            if url not in self.meta_data:
                continue
            f = None
            txt_list = os.listdir(os.path.join(self.dataset_path, file))
            
            if data_type == "[NOHASH]":
                if len(txt_list) == 1 and txt_list[0].endswith(".txt"):
                    f = open(os.path.join(self.dataset_path, file, txt_list[0]), "r", errors="ignore")
                    data, site_sl = self.set_data(f)
                    if data is None:
                        continue
                    f.close()
                else:
                    try:
                        data = {}
                        site_sl = 10
                        if len(txt_list) == 1:
                            _file = txt_list[0]
                            txt_list = os.listdir(os.path.join(self.dataset_path, file, _file))
                        f_list = self.select_txt(txt_list, data_type=data_type)
                        if len(f_list) == 0:
                            continue
                        for _f in f_list:
                            f = open(os.path.join(self.dataset_path, file, _file,_f), "r", errors="ignore")
                            _data, _site_sl = self.set_data(f)
                            if _data is None:
                                continue
                            data.update(_data)
                            if _site_sl < site_sl:
                                site_sl = _site_sl
                            f.close()
                    except:
                        continue
                    
            elif data_type == "[HASH+NOHASH]":
                if len(txt_list) == 1 and txt_list[0].endswith(".txt"):
                    continue
                else:
                    try:
                        data = {}
                        site_sl = 10
                        folder_path = os.path.join(self.dataset_path, file)
                        if len(txt_list) == 1:
                            _file = txt_list[0]
                            txt_list = os.listdir(os.path.join(self.dataset_path, file, _file))
                            folder_path = os.path.join(folder_path, _file)
                        f_list = self.select_txt(txt_list, data_type=data_type)
                        if len(f_list) == 0:
                            continue
                        for _f in f_list:
                            f = open(os.path.join(folder_path, _f), "r", errors="ignore")
                            _data, _site_sl = self.set_data(f)
                            if _data is None:
                                continue
                            data.update(_data)
                            if _site_sl < site_sl:
                                site_sl = _site_sl
                            f.close()
                    except:
                        continue
            elif data_type == "[HASH]":
                del self.meta_data[url]
                continue
            else:
                del self.meta_data[url]
                continue
            if site_sl == 10:
                del self.meta_data[url]
                continue
            with open(os.path.join(self.site_path, url + ".json"), "w") as f:
                json.dump(data, f, indent=4)
            
            self.meta_data[url]["sl"] = site_sl
            self.logger.print(f"[*] Processing {i:5}/{site_num}")
        self.logger.print(f"[+] Done processing sites")
        self.logger.print(f"[+] Sites: {len(self.meta_data)}")
            
            
    def set_data(self, f):
        data = {}
        site_sl = 10
        if f is None:
            return None, None
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if ":" not in line:
                continue
            split = line.split(":")
            if len(split) != 2:
                if split[1] ==  split[2]:
                    id, pwd = split[0], split[1]
                elif "::" in line:
                    split = line.split("::")
                    if len(split) == 2:
                        id, pwd = split[0], split[1]
                    else:
                        continue
                else:
                    continue
            else:
                id, pwd = line.split(":", 1)
            if id.strip() == "" or pwd.strip() == "":
                continue
            pwd_sl = check_pwd_security_level(pwd.strip())
            if pwd_sl < site_sl:
                site_sl = pwd_sl
            data[id.strip()] = pwd.strip()
        return data, site_sl
    
    
    def select_txt(self, txt_list, data_type="[NOHASH]"):
        f_list = []
        if len(txt_list) == 1 and data_type == "[NOHASH]":
            f_list.append(txt_list[0])
        else:
            for txt in txt_list:
                if "decrypted" in txt or "no hash" in txt or ("[NOHASH]" in txt and "[HASH]" not in txt) or "Result" in txt or "Rejected" in txt or "[nohash]" in txt or "good" in txt:
                    f_list.append(txt)
        return f_list
        
        
        
def init():
    parser = argparse.ArgumentParser(description="Parser for data processing configuration")
    parser.add_argument('--run_type', type=str, default="data_processing", help='run type: train/test/data_processing')
    parser.add_argument('--feature', type=str, default="all", help="dataset feature: all, no_html")
    return parser.parse_args()

if __name__ == "__main__":
    args = init()
    logger = Logger(args)
    processor = DataProcessor(args, logger)
    processor.load_meta_data()
    processor.set_sl()
    processor.save_meta_data()