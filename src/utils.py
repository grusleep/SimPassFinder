import re
import torch



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