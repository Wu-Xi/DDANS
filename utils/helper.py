'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re
import pickle
import datetime
import torch
import json

def save_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur1 = cur.strftime('%b-%d-%Y_%H-%M-%S')
    cur2 = cur.strftime('%b-%d-%Y')

    return (cur1,cur2)

def save_checkpoint(args, epoch, best_valid_score, model, saved_model_file):
    r"""Store the model parameters information and training information.

    Args:
        args: 
        epoch (int): the current epoch id
        best_valid_score:
        model:
        saved_model_file: 

    """
    print("Save parameters of the model")
    state = {
        'config': args,
        'epoch': epoch,
        'best_valid_score': best_valid_score,
        'state_dict': model.state_dict()
    }
    torch.save(state, saved_model_file)

def save_list_to_file(lst, filename):
    with open(filename, "w") as file:
        file.write(json.dumps(lst))

def read_file_to_list(filename):
    with open(filename, "r") as file:
        data = file.read()
    return json.loads(data)
