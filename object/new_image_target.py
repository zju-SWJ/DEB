import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import img_network, img_loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import time
np.set_printoptions(threshold=np.inf)
import shutil
from tensorboardX import SummaryWriter

def setDir(filepath):
    if not osp.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
        
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        #normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        #normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, delete=args.delete, fill=args.fill, seed=0, transform=image_train())
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["target_idx"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders, dsets["target"]
'''
def sample_by_entropy(cur_entropy_list, gen_num):
    gen_list = []
    maxx = 0
    cumulative = []
    cur_entropy_list = np.power(cur_entropy_list, 1)
    for i in range(len(cur_entropy_list)):
        maxx += cur_entropy_list[i]
        temp = copy.deepcopy(maxx)
        cumulative.append(temp)
    x = np.random.uniform(0, maxx, gen_num)
    for i in range(gen_num):
        for j in range(len(cumulative)):
            if x[i] <= cumulative[j]:
                gen_list.append(j)
                break
    return gen_list
'''
def sample_by_entropy_threshold(cur_entropy_list, gen_num, threshold):
    low_entropy_idx_list = []
    for i in range(len(cur_entropy_list)):
        if cur_entropy_list[i] <= threshold:
            low_entropy_idx_list.append(i)
    if len(low_entropy_idx_list) == 0:
        print("CurrentThreshold:", threshold)
        print("CurrentEntropyList:", cur_entropy_list)
    select_item_idx = np.random.randint(0, high = len(low_entropy_idx_list), size = gen_num)
    gen_list = []
    for i in range(gen_num):
        gen_list.append(low_entropy_idx_list[select_item_idx[i]])
    return gen_list

#'''
def sample_by_entropy_window(cur_entropy_list, gen_num, min_window, max_window):
    entropy_idx_list = []
    for i in range(len(cur_entropy_list)):
        if cur_entropy_list[i] <= max_window and cur_entropy_list[i] >= min_window:
            entropy_idx_list.append(i)
    if len(entropy_idx_list) == 0:
        print("CurrentMinWindow:", min_window)
        print("CurrentMaxWindow:", max_window)
        print("CurrentEntropyList:", cur_entropy_list)
    select_item_idx = np.random.randint(0, high = len(entropy_idx_list), size = gen_num)
    gen_list = []
    for i in range(gen_num):
        gen_list.append(entropy_idx_list[select_item_idx[i]])
    return gen_list
#'''

def balanced_digit_load(args, train_target, entropy_list, class_list, epoch_num, percent, each_class_num):
    if percent == -1:
        return {}, []
    gen_list = []
    count_list = []
    idx_matrix = []
    for i in range(args.class_num):
        count_list.append(0)
        temp = []
        idx_matrix.append(temp)
    for i in range(len(class_list)):
        count_list[class_list[i]] += 1
        idx_matrix[class_list[i]].append(i)
        gen_list.append(1)
        
    class_reliable_entropy_list = []
    class_reliable_entropy_list_low = []
    for i in range(args.class_num):
        pos = int(percent * len(idx_matrix[i]))
        if pos == 0:
            pos = 1
        cur_entropy_l = copy.deepcopy(entropy_list[idx_matrix[i]])
        if len(cur_entropy_l) == 0:
            class_reliable_entropy_list.append(0)
            continue
        cur_entropy_l.sort()
        max_reliable_entropy = cur_entropy_l[pos - 1]
        class_reliable_entropy_list.append(max_reliable_entropy)
    
    if args.window_size > 0:
        for i in range(args.class_num):
            min_window = max(0.0, percent - args.window_size)
            pos_ = int(min_window * len(idx_matrix[i]))
            if pos_ == 0:
                pos_ = 1
            cur_entropy_l_low = copy.deepcopy(entropy_list[idx_matrix[i]])
            if len(cur_entropy_l_low) == 0:
                class_reliable_entropy_list_low.append(1)
                continue
            cur_entropy_l_low.sort()
            max_reliable_entropy_ = cur_entropy_l_low[pos_ - 1]
            class_reliable_entropy_list_low.append(max_reliable_entropy_)

    max_threshold = np.max(class_reliable_entropy_list)
    
    predict_class_num = each_class_num.numpy().astype(int).tolist()
    for i in range(args.class_num):
        gen_num = 0
        if args.naive_dis == True:
            gen_num = np.max(count_list) - count_list[i]
        else:
            gen_num = np.max(predict_class_num) - predict_class_num[i]
        if args.gen_speed != 'max':
            if args.gen_speed == 'linear':
                gen_num = int(gen_num * (0.5 + 0.5 * np.power(epoch_num / (args.max_epoch - 1.0), 1)))
            elif args.gen_speed == 'faster':
                gen_num = int(gen_num * (0.5 + 0.5 * np.power(epoch_num / (args.max_epoch - 1.0), 2)))
            elif args.gen_speed == 'slower':
                gen_num = int(gen_num * (0.5 + 0.5 * np.power(epoch_num / (args.max_epoch - 1.0), 0.5)))
        if count_list[i] == 0 or gen_num == 0:
            continue
        if args.naive_sample == True:
            select_item_idx = np.random.randint(0, high = count_list[i], size = gen_num)
        else:
            if args.window_size > 0:
                select_item_idx = sample_by_entropy_window(copy.deepcopy(entropy_list[idx_matrix[i]]), gen_num, np.min(class_reliable_entropy_list_low), max_threshold)
            else:
                select_item_idx = sample_by_entropy_threshold(copy.deepcopy(entropy_list[idx_matrix[i]]), gen_num, max_threshold)
        for j in range(gen_num):
            gen_list[idx_matrix[i][select_item_idx[j]]] += 1
        result = copy.deepcopy(train_target.imgs)[idx_matrix[i]][select_item_idx]
        train_target.imgs = np.concatenate((train_target.imgs, result), axis = 0)

    balanced_dset_loaders = {}
    balanced_dset_loaders["target"] = DataLoader(train_target, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    balanced_dset_loaders["target_idx"] = DataLoader(train_target, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    return balanced_dset_loaders, gen_list

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(img_loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    recall = matrix.diagonal() / matrix.sum(axis=1) * 100
    precision = matrix.diagonal() / matrix.sum(axis=0) * 100
    ave_recall = recall.mean()
    aa = [str(np.round(i, 2)) for i in recall]
    bb = [str(np.round(i, 2)) for i in precision]
    class_recall = ' '.join(aa)
    class_precision = ' '.join(bb)
    return ave_recall, class_recall, accuracy*100, mean_ent, class_precision

def obtain_label(loader, netF, netB, netC, args, weight):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    
    all_fea = all_fea.float().cpu().numpy()
    aff = all_output.float().cpu().numpy()
    K = all_output.size(1)
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.cls_threshold)
    labelset = labelset[0]
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Without/With Pseudo Labeling Target Training Accuracy = {:.2f}%/{:.2f}%'.format(accuracy*100, acc*100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    #print(log_str+'\n')

    return pred_label.astype('int')

def train_target(args, output_dir_temp):
    writer = SummaryWriter(log_dir = output_dir_temp)
    dset_loaders, train_tar = data_load(args)
    copy_threshold = copy.deepcopy(args.threshold)
    acc_list = []
    accuracy_list = []
    ## set base network
    if args.net[0:3] == 'res':
        netF = img_network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = img_network.VGGBase(vgg_name=args.net).cuda()  

    netB = img_network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = img_network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_epoch = args.max_epoch
    epoch_num = 0
    entropy_list = []
    class_list = []
    balanced_dset_loaders, weight = balanced_digit_load(args, copy.deepcopy(train_tar), entropy_list, class_list, 0, -1, 0)
    while epoch_num < max_epoch:
        total_time = time.time()
        optimizer.zero_grad()        
        netF.eval()
        netB.eval()
        start_test = True
        with torch.no_grad():
            iterr = iter(dset_loaders['target_idx'])
            for _ in range(len(dset_loaders['target_idx'])):
                data = iterr.next()
                inputs = data[0]
                inputs = inputs.cuda()
                feas = netB(netF(inputs))
                outputs = netC(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        if args.balance == True:
            entropy = img_loss.Entropy(all_output)
            each_class_num = torch.sum(all_output, 0)
            entropy = preprocessing.MinMaxScaler().fit_transform(entropy.reshape(-1, 1))
            entropy_list = copy.deepcopy(entropy.reshape(1, -1)[0])
            class_list = copy.deepcopy(predict.numpy().tolist())
            balanced_dset_loaders, weight = balanced_digit_load(args, copy.deepcopy(train_tar), entropy_list, class_list, epoch_num, copy_threshold, each_class_num)
            weight = np.array(weight)
        else:
            count_l = [0] * args.class_num
            copy_predict = copy.deepcopy(predict.numpy().tolist())
            for i in range(len(copy_predict)):
                count_l[copy_predict[i]] += 1
            # print('Model Predict Distribution:', count_l)
        if args.cls_par > 0:
            if args.balance == True:
                mem_label = obtain_label(balanced_dset_loaders['target_idx'], netF, netB, netC, args, weight)
            else:
                mem_label = obtain_label(dset_loaders['target_idx'], netF, netB, netC, args, weight)
            mem_label = torch.from_numpy(mem_label).cuda()
        netF.train()
        netB.train()
        if args.balance == True:
            iter_len = len(balanced_dset_loaders["target"])
        else:
            iter_len = len(dset_loaders["target"])
        for i in range(iter_len):    
            try:
                inputs_test, _, tar_idx = iter_test.next()
            except:
                if args.balance == True:
                    iter_test = iter(balanced_dset_loaders["target"])
                else:
                    iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = iter_test.next() #one batch
            if inputs_test.size(0) == 1:
                continue
            lr_scheduler(optimizer, iter_num = epoch_num * iter_len + i + 1, max_iter = max_epoch * iter_len)
            # lr_scheduler(optimizer, iter_num = epoch_num * iter_len + i, max_iter = max_epoch * iter_len - 1)
            inputs_test = inputs_test.cuda()
            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)
            if args.cls_par > 0:
                pred = mem_label[tar_idx]
                classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
                if epoch_num == 0 and args.dset == "VISDA-C":
                    classifier_loss *= 0
            else:
                classifier_loss = torch.Tensor(0.0).cuda()
            if args.ent == True:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(img_loss.Entropy(softmax_out))
                if args.gent == True:
                    msoftmax = softmax_out.mean(dim=0)
                    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                im_loss = entropy_loss * args.ent_par
                classifier_loss += im_loss
            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

        netF.eval()
        netB.eval()
        acc, recall, accuracy, _, precision = cal_acc(dset_loaders['test'], netF, netB, netC)
        # print('Class Recall:', recall)
        # print('Class Precision:', precision)
        acc_list.append(acc)
        accuracy_list.append(accuracy)
        writer.add_scalar(args.net + '/Acc', accuracy, epoch_num)
        writer.add_scalar(args.net + '/Recall', acc, epoch_num)

        log_str = 'Training Target Iter: {}/{}; Test Acc = {:.2f}/{:.2f}%'.format(epoch_num + 1, max_epoch, acc, accuracy)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str)
        # print('Total Time:', time.time() - total_time, 's')
        epoch_num += 1

        if args.threshold_speed != "min":
            if args.threshold_speed == 'linear':
                copy_threshold = args.threshold + (args.max_threshold - args.threshold) * np.power((epoch_num / (max_epoch - 1.0)), 1)
            elif args.threshold_speed == 'slower':
                copy_threshold = args.threshold + (args.max_threshold - args.threshold) * np.power((epoch_num / (max_epoch - 1.0)), 0.5)
            elif args.threshold_speed == 'faster':
                copy_threshold = args.threshold + (args.max_threshold - args.threshold) * np.power((epoch_num / (max_epoch - 1.0)), 2)

    if args.issave == True:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    log_str = 'Max Acc = {:.2f}/{:.2f}%'.format(np.max(acc_list), np.max(accuracy_list))
    print(log_str)
    writer.close()
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def str2bool(str):
    return True if str.lower() == 'true' else False

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    # parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA-C', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="alexnet, vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
 
    parser.add_argument('--gent', type=str2bool, default='True')
    parser.add_argument('--ent', type=str2bool, default='True')
    parser.add_argument('--cls_threshold', type=int, default=-1)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='tar_result')
    parser.add_argument('--output_src', type=str, default='src_result')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=str2bool, default='False')

    parser.add_argument('--fill', type=str2bool, default='False')
    parser.add_argument('--delete', type=str2bool, default='True')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--balance', type=str2bool, default='False')
    parser.add_argument('--naive_dis', type=str2bool, default='False')
    parser.add_argument('--naive_sample', type=str2bool, default='False')
    parser.add_argument('--max_threshold', type=float, default=1)
    parser.add_argument('--threshold_speed', type=str, default="min", choices=["linear", "slower", "faster", "min"])
    parser.add_argument('--gen_speed', type=str, default="max", choices=["linear", "slower", "faster", "max"])
    parser.add_argument('--window_size', type=float, default=0)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    setup_seed(args.seed)

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        # print(i)
        folder = os.getcwd() + '/data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper(), args.net)
        args.output_dir = osp.join(args.output, args.net + '_seed' + str(args.seed))
        # args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = ''
        if args.balance == True:
            if args.naive_dis == True and args.naive_sample == True:#NN
                args.savename += 'NN'
            if args.naive_dis == True and args.naive_sample == False:#NP
                args.savename += 'NP'
                if args.threshold_speed == 'min':
                    args.savename += str(args.threshold)
                else:
                    args.savename += args.threshold_speed + str(args.threshold) + 'to' + str(args.max_threshold)
                if args.window_size > 0:
                    args.savename += '_window' + str(args.window_size)
            if args.naive_dis == False and args.naive_sample == True:#PN
                args.savename += 'PN'
            if args.naive_dis == False and args.naive_sample == False:#PP
                args.savename += 'PP'
                if args.threshold_speed == 'min':
                    args.savename += str(args.threshold)
                else:
                    args.savename += args.threshold_speed + str(args.threshold) + 'to' + str(args.max_threshold)     
                if args.window_size > 0:
                    args.savename += '_window' + str(args.window_size)
        else:
            args.savename += 'SHOT'

        output_dir_temp = osp.join(args.output_dir, args.savename)
        if not osp.exists(output_dir_temp):
            os.system('mkdir -p ' + output_dir_temp)
        if not osp.exists(output_dir_temp):
            os.mkdir(output_dir_temp)
        args.out_file = open(osp.join(output_dir_temp, 'log_tar.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        print('Task: ', args.savename)
        train_target(args, output_dir_temp)