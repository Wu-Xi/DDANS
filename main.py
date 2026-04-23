import os
import random
import torch
import numpy as np
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping, get_local_time

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling_origin(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            _observed_pos_list = train_set[user]
            observed_pos_set = set(_observed_pos_list)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in observed_pos_set:
                        break
                negitems.append(negitem)
                           
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling_origin(entity_pairs, train_pos_set, n_negs)).to(device)  
    
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, item_group_idx = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']

    """define model"""
    from modules.LightGCN import LightGCN 
    model = LightGCN(n_params, args, norm_mat).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print(f"{get_local_time()[0]} start training ...")
    test_ret_list = []
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_ , user_dict['train_user_set'], s , s + args.batch_size, args.n_negs)
            batch_loss,bpr_loss,reg_loss= model(batch,epoch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            s += args.batch_size

        train_e_t = time()
        if epoch % 1 == 0:
            """testing"""
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, mode='test')
            test_e_t = time()
            test_result = [epoch, int(train_e_t - train_s_t), int(test_e_t - test_s_t), round(loss.item(), 2), test_ret['recall'], test_ret['ndcg'], test_ret['precision'], test_ret['hit_ratio']]
            test_ret_list.append(test_result)
            train_res.add_row(test_result)

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, int(train_e_t - train_s_t), int(test_e_t - test_s_t), round(loss.item(), 2), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['hit_ratio']])
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][1], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=10)
            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][1] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')
        else:
            print('using time %.4fs, training loss at epoch %d: %.4f' % (int(train_e_t - train_s_t), epoch, round(loss.item(), 2)))

    print(f"{get_local_time()[0]} end training ...")
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    test_ret = test_ret_list[-11]
    train_res.clear_rows()
    train_res.add_row(test_ret)
    print(train_res)
    