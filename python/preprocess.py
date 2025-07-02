import os
import json
import shutil
import socket
import rarfile
import requests
from urllib.parse import urlparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src import *



dataset_path = "dataset"
dataset_types = ["NOHASH", "HASH+NOHASH"]

def process_file(args):
    dataset_type, file = args
    error_files = 0
    categorys = set()
    all_files = 1

    if not file.endswith(".rar"):
        return (categorys, error_files, all_files)

    site = file.split(" ")[0]
    index = file.find(f"[{dataset_type}]")
    if dataset_type == "NOHASH":
        temp = file[index+9:]
    else:
        temp = file[index+14:]
    index = temp.find(")")
    category = temp[1:index]

    if "xt" in category:
        category = category[4:]
    categorys.add(category)

    try:
        _site = "https://" + site
        parsed_url = urlparse(_site)
        hostname = parsed_url.hostname
        ip_address = socket.gethostbyname(hostname)
    except:
        ip_address = None

    try:
        with rarfile.RarFile(os.path.join(dataset_path, dataset_type, file)) as rf:
            extract_path = f"dataset/extract/{dataset_type}/{site}_{category}"
            if ip_address:
                extract_path += f"_{ip_address}"
            rf.extractall(extract_path)
    except:
        error_files = 1

    return (categorys, error_files, all_files)


def extract_dataset():
    categorys = set()
    error_files = 0
    all_files = 0
    pool_args = []

    for dataset_type in dataset_types:
        file_list = os.listdir(os.path.join(dataset_path, dataset_type))
        pool_args.extend([(dataset_type, file) for file in file_list])

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, pool_args), total=len(pool_args), desc="[*] Extracting files"))

    for cats, err, count in results:
        categorys.update(cats)
        error_files += err
        all_files += count

    print("[*] Done processing all folders")
    print(f"[*] Error files: {error_files}/{all_files}")

    with open("categorys.json", "w") as f:
        json.dump(list(categorys), f, indent=4, ensure_ascii=False)
        
        
def check_file():
    for dataset_type in dataset_types:
        txt_files = 0
        txts_files = 0
        decrypted_txt = 0
        no_hash_txt = 0
        txts_list = {}
        file_list = os.listdir(os.path.join(dataset_path,"extract", dataset_type))
        for file in tqdm(file_list, desc=f"[*] Checking {dataset_type}"):
            if os.path.isdir(os.path.join(dataset_path, "extract", dataset_type, file)):
                txt_list = os.listdir(os.path.join(dataset_path, "extract", dataset_type, file))
                if len(txt_list) == 1:
                    txt_files += 1
                elif len(txt_list) > 1:
                    txts_list[file] = txt_list
                    txts_files += 1
                    for txt in txt_list:
                        if "decrypted" in txt:
                            decrypted_txt += 1
                        elif "no hash" in txt:
                            no_hash_txt += 1
                            
                            
        print(f"[*] txt_files: {txt_files}")
        print(f"[*] txts_files: {txts_files}")
        print(f"[*] decrypted: {decrypted_txt}/{txts_files}")
        print(f"[*] no_hash: {no_hash_txt}/{txts_files}")
        with open(f"{dataset_type}_txts.json", "w") as f:
            json.dump(txts_list, f, indent=4, ensure_ascii=False)
            
    
def select_NOHASH():
    file_list = os.listdir(os.path.join(dataset_path, "extract", "NOHASH"))
    select_file = {}
    for file in tqdm(file_list, desc="[*] Checking NOHASH"):
        txt_list = os.listdir(os.path.join(dataset_path, "extract", "NOHASH", file))
        if len(txt_list) == 1:
            select_file[file] = txt_list
        if len(txt_list) > 1:
            try:
                data_len_list = []
                for txt in txt_list:
                        data_len = txt.split(" ")[1]
                        data_len_list.append(int(data_len[1:-1].replace(".", "")))
                min_len = min(data_len_list)
                min_index = data_len_list.index(min_len)
                select_file[file] = txt_list[min_index]
            except:
                pass
    with open("select_NOHASH.json", "w") as f:
        json.dump(select_file, f, indent=4, ensure_ascii=False)
        
    
def select_HASH_NOHASH():
    file_list = os.listdir(os.path.join(dataset_path, "extract", "HASH+NOHASH"))
    select_file = {}
    len_list = {}
    txts_num = 0
    has_decrypt = 0
    has_nohash = 0
    has_result = 0
    has_good = 0
    for file in tqdm(file_list, desc="[*] Checking HASH+NOHASH"):
        txt_list = os.listdir(os.path.join(dataset_path, "extract", "HASH+NOHASH", file))
        if len(txt_list) == 1:
            if txt_list[0].endswith(".txt"):
                select_file[file] = txt_list
        if len(txt_list) > 1:
            txts_num += 1
            select_file[file] = []
            for txt in txt_list:
                if "decrypted" in txt:
                    select_file[file].append(txt)
                elif "no hash" in txt:
                    select_file[file].append(txt)
                elif txt == "Result.txt" or txt == "result.txt":
                    select_file[file].append(txt)
                elif txt == "good.txt":
                    select_file[file].append(txt)
    with open("select_HASH+NOHASH.json", "w") as f:
        json.dump(select_file, f, indent=4, ensure_ascii=False)

def get_data():
    meta_data = {}
    data = {}
    
    for data_type in dataset_types:
        with open(f"json/select_{data_type}.json", "r") as f:
            json_data = json.load(f)
            
        for key, value in tqdm(json_data.items(), desc=f"[*] Processing {data_type}"):
            site = {}
            if len(key.split("_")) == 2:
                site_name, category = key.split("_")
                ip = None
            elif len(key.split("_")) == 3:
                site_name, category, ip = key.split("_")
            else:
                continue
            site["category"] = category
            site["ip"] = ip
            
            if site_name not in meta_data:
                meta_data[site_name] = site
                data[site_name] = {}
            
            for txt in value:
                if txt.endswith(".txt"):
                    
                    if not os.path.exists(os.path.join(dataset_path, "extract", data_type, key, txt)):
                        print(os.path.join(dataset_path, "extract", data_type, key, txt))
                    with open(os.path.join(dataset_path, "extract", data_type, key, txt), "r", encoding="latin1") as f:
                        for line in f:
                            line = line.strip()
                            if line == "":
                                continue
                            if ":" not in line:
                                continue
                            id, pwd = line.split(":", 1)
                            if id in data[site_name]:
                                continue
                            data[site_name][id] = pwd
                            
    for site in tqdm(data, desc="[*] Calculating security level"):
        site_security_level = 0
        for id in data[site]:
            pwd = data[site][id]
            pwd_security_level = check_pwd_security_level(pwd)
            if pwd_security_level > site_security_level:
                site_security_level = pwd_security_level
        meta_data[site]["security_level"] = site_security_level
    print("[+] Done calculating security level")
    
    with open(os.path.join(dataset_path, "meta_data", "all_meta_data.json"), "w") as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)
    for site in tqdm(data, desc="[*] Saving data"):
        with open(os.path.join(dataset_path, "sites", f"{site}.json"), "w") as f:
            json.dump(data[site], f, indent=4, ensure_ascii=False)
    print("[+] Done saving data")
    
    
def preprocess_meta_data():
    with open(os.path.join(dataset_path, "meta_data","all_meta_data.json"), "r") as f:
        meta_data = json.load(f)
    with open(os.path.join(dataset_path, "categorys.json"), "r") as f:
        categorys = json.load(f)
    for site in tqdm(meta_data, desc="[*] Preprocessing meta data"):
        for key, value in categorys.items():
            if meta_data[site]["category"] in value:
                meta_data[site]["category"] = key
                break
    with open(os.path.join(dataset_path, "new_meta_data.json"), "w") as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)
    print("[+] Done preprocessing meta data")
    
          
def set_ip():
    site_has_ip = 0
    with open(os.path.join(dataset_path, "meta_data","all_meta_data.json"), "r") as f:
        meta_datas = json.load(f)
    for site in tqdm(meta_datas, desc="[*] Setting ip"):
        meta_data = meta_datas[site]
        if meta_data["ip"] is not None:
            site_has_ip += 1
    print(f"[+] Site has ip: {site_has_ip}")


def check_ip():
    with open(os.path.join(dataset_path, "meta_data","all_meta_data.json"), "r") as f:
        meta_data = json.load(f)
    temp = {}
    for site in tqdm(meta_data, desc="[*] Checking ip"):
        ip = meta_data[site]["ip"]
        if ip is not None:
            temp[site] = meta_data[site]
    with open(os.path.join(dataset_path, "meta_data","meta_data.json"), "w") as f:
        json.dump(temp, f, indent=4, ensure_ascii=False)
    print("[+] Done getting ip")
    

def set_country():
    with open(os.path.join(dataset_path, "meta_data", "meta_data.json"), "r") as f:
        meta_data = json.load(f)
    new_meta_data = {}
    num = 0
    for site in tqdm(meta_data, desc="[*] Setting country"):
        site_data = meta_data[site]
        ip = site_data["ip"]
        url = f"https://ipinfo.io/{ip}/json?token=3b5727e87d6c6b"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            try:
                country = data["country"]
                site_data["country"] = country
                new_meta_data[site] = site_data
                num += 1
            except:
                continue
    with open(os.path.join(dataset_path, "meta_data", "meta_data.json"), "w") as f:
        json.dump(new_meta_data, f, indent=4, ensure_ascii=False)
    print(f"[+] Done setting country: {num}")
    
    
def check_security_level():
    with open(os.path.join(dataset_path, "meta_data", "meta_data.json"), "r") as f:
        meta_data = json.load(f)
    for site in tqdm(meta_data, desc="[*] Checking security level"):
        with open(os.path.join(dataset_path, "sites", f"{site}.json"), "r") as f:
            site_data = json.load(f)
        for id, pwd in site_data.items():
            pwd_security_level = check_pwd_security_level(pwd)
            if pwd_security_level < meta_data[site]["security_level"]:
                meta_data[site]["security_level"] = pwd_security_level
    
    with open(os.path.join(dataset_path, "meta_data", "meta_data.json"), "w") as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)
    print("[+] Done checking security level")
    
    
def extract_users():
    path = os.path.join(dataset_path, "users")
    file_list = os.listdir(path)
    nodes = []
    for file in tqdm(file_list, desc="[*] Extracting users"):
        with open(os.path.join(path, file), "r") as f:
            user_data = json.load(f)
        for node in user_data:
            nodes.append(node)
    with open(os.path.join(dataset_path, "nodes_users.json"), "w") as f:
        json.dump(nodes, f, indent=4, ensure_ascii=False)
    print("[+] Done extracting users")
    
    
def collection_get_meta_data():
    num = 0
    folder_list = os.listdir(os.path.join(dataset_path, "collection"))
    meta_data = {}
    for folder in folder_list:
        folder_path = os.path.join(dataset_path, "collection", folder)
        file_list = os.listdir(folder_path)
        for file in tqdm(file_list, desc=f"[*] Processing {folder}"):
            if file.endswith(".txt"):
                data = {}
                check = True
                if "[HASH+NOHASH]" in file:
                    continue
                if "[NOHASH]" in file:
                    site = file.split(" ")[0]
                else:
                    site = file[:-4]
                if site in meta_data:
                    continue
                if site == "":
                    continue
                meta_data[site] = {}
                if folder == "game":
                    meta_data[site]["category"] = "Entertainment"
                else:
                    meta_data[site]["category"] = "NoCategory"
                if len(site) > 63:
                    continue
                try:
                    ip_address = socket.gethostbyname(site)
                except socket.gaierror as e:
                    ip_address = None
                    check = False
                meta_data[site]["ip"] = ip_address
                if ip_address is None:
                    continue
                try:
                    url = f"https://ipinfo.io/{ip_address}/json?token=3b5727e87d6c6b"
                    response = requests.get(url)
                    if response.status_code == 200:
                        ip_data = response.json()
                        try:
                            country = ip_data["country"]
                            meta_data[site]["country"] = country
                            num+=1
                        except:
                            check = False
                except:
                    check = False
                        
                if check:        
                    with open(os.path.join(folder_path, file), "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                                line = line.strip()
                                if line == "":
                                    continue
                                if ":" not in line:
                                    continue
                                id, pwd = line.split(":", 1)
                                if id in data:
                                    continue
                                data[id] = pwd
                    
                    site_security_level = 10
                    for id in data:
                        pwd = data[id]
                        pwd_security_level = check_pwd_security_level(pwd)
                        if pwd_security_level < site_security_level:
                            site_security_level = pwd_security_level
                    meta_data[site]["security_level"] = site_security_level
                
                    with open(os.path.join(dataset_path, "sites", f"{site}.json"), "w") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
    with open(os.path.join(dataset_path, "meta_data", "collection_meta_data.json"), "w") as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)
    print(f"[+] Done getting meta data: {num}")
                    
                    

if __name__ == "__main__":
    with open(os.path.join(dataset_path, "graph", "edges.json"), "r") as f:
        data = json.load(f)
    sorted(data, key=lambda x: (x['node_1'], x['node_2']))
    with open(os.path.join(dataset_path, "graph", "edges.json"), "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    