import re
import os
import sys
import torch
from tqdm import tqdm as _tqdm


def check_pwd_security_level(pwd: list):
    level = 0
    if len(pwd) >= 12:
        level += 1
    if re.search(r'[0-9]', pwd):
        level += 1
    if re.search(r'[a-z]', pwd):
        level += 1
    if re.search(r'[A-Z]', pwd):
        level += 1
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', pwd):
        level += 1
    return level


def save_checkpoint(path, model, optimizer, valid_result):
    if path == None:
        return
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_result': valid_result
    }, path)
    

def load_checkpoint(path, model, optimizer):
    if path == None:
        return
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['valid_result']


def tqdm(*args, **kwargs):
    return _tqdm(*args, file=sys.stderr, **kwargs)



class Logger:
    def __init__(self, args):
        if args.run_type in ["train", "test"]:
            self.model_name = args.model_name
            print(self.model_name)
            if args.run_type == "train":
                self.log_path = os.path.join("../log", args.setting, f"{self.model_name}.log")
            else:
                self.log_path = os.path.join("../log", args.setting, f"{self.model_name}_test.log")
        else:
            self.log_path = os.path.join("../log", f"{args.run_type}.log")
        self.reset_log()
        
        
    def reset_log(self):
        with open(self.log_path, 'w') as f:
            f.write("")
        
        
    def print(self, msg):
        print(msg)
        with open(self.log_path, 'a') as f:
            f.write(msg + "\n")
    
    