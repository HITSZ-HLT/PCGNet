import argparse
import numpy as np
import datetime
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import IEMOCAPDataset, MELDDataset
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PCGNet import GATModel
from loss import FocalLoss
from loss import create_class_weight_SCL, CKSCL
from torch.optim import Adam
import itertools



def decodeShiftLabel(shift_label, dataset):
    if dataset == 'IEMOCAP':
        class_num = 6
    else:
        class_num = 7
    front_label = int(shift_label // class_num)
    back_label = int(shift_label % class_num)
    return front_label, back_label



def shiftContrastLearning(shift_features, shift_labels, sample_num=50, dataset='IEMOCAP'):

    shift_nums = shift_features.shape[0]
    sample_num = sample_num if shift_nums > sample_num else shift_nums
    sample_index = torch.tensor(random.sample(list(range(shift_nums)), sample_num)).detach()
    sample_index = sample_index.to(shift_features.device)

    shift_features_sampled = shift_features[sample_index]
    shift_labels_sampled = shift_labels[sample_index]

    ### TODO : improve sample stragy
    shift_features_pair_index = list(itertools.permutations(list(range(sample_num)), 2))

    shift_features_pair = []
    shift_features_pair_label = []
    for index in shift_features_pair_index:
        index = list(index)
        shift_features_pair.append(shift_features_sampled[index,:].view(-1))
        pair1_front, pair1_back = decodeShiftLabel(int(shift_labels_sampled[index[0]]), dataset=dataset)
        pair2_front, pair2_back = decodeShiftLabel(int(shift_labels_sampled[index[1]]), dataset=dataset)

        ## 标签相反
        if pair1_front == pair2_back and pair1_back == pair2_front:
            shift_features_pair_label.append(4)
        else:
            ## 标签相同
            if pair1_front == pair2_front and pair1_back == pair2_back:
                shift_features_pair_label.append(0)
            ## 头相同
            elif pair1_front == pair2_front and pair1_back != pair2_back:
                shift_features_pair_label.append(1)
            ## 尾相同
            elif pair1_front != pair2_front and pair1_back == pair2_back:
                shift_features_pair_label.append(2)
            ## 不相同
            elif pair1_front != pair2_front and pair1_back != pair2_back:
                shift_features_pair_label.append(3)
    return torch.stack(shift_features_pair), torch.tensor(shift_features_pair_label).to(shift_features.device)










def metricsIntraShift(labels, preds, speakers, dataset):
    if dataset == "IEMOCAP":
        label_sum = 6
    elif dataset == "MELD":
        label_sum = 7
    bias = 0
    speaker_values = {}
    for speaker in speakers:
        cur_label = labels[bias:bias+len(speaker)]
        cur_pred = preds[bias:bias+len(speaker)]
        bias = bias + len(speaker)
        speaker = np.array(speaker)
        speaker_set = set(speaker)
        for s in speaker_set:
            if s not in speaker_values.keys():
                speaker_values[s] = []
            indices = np.where(speaker == s)[0]
            speaker_values[s].append((cur_label[indices], cur_pred[indices]))
        
    shift_labels = []
    shift_preds = []
    for s in speaker_values.keys():
        value = speaker_values[s]
        value_shift_label = []
        value_shift_pred = []
        for item in value:
            value_shift_label.append(label_sum*item[0][:-1] + item[0][1:])
            value_shift_pred.append(label_sum*item[1][:-1] + item[1][1:])
        shift_labels.append(np.concatenate(value_shift_label))
        shift_preds.append(np.concatenate(value_shift_pred))
    shift_labels = np.concatenate(shift_labels)
    shift_preds = np.concatenate(shift_preds)

    # print(metrics.classification_report(shift_labels, shift_preds, digits=4))
    shift_micro_f1 = metrics.f1_score(shift_labels, shift_preds, average='micro')
    shift_macro_f1 = metrics.f1_score(shift_labels, shift_preds, average='macro')
    shift_weighted_f1 = metrics.f1_score(shift_labels, shift_preds, average='weighted')
    shift_acc = metrics.accuracy_score(shift_labels, shift_preds)

    # print("==== shift_micro_f1 = {} ====".format(shift_micro_f1))
    # print("==== shift_macro_f1 = {} ====".format(shift_macro_f1))
    # print("==== shift_weighted_f1 = {} ====".format(shift_weighted_f1))
    # print("==== shift_acc = {} ====".format(shift_acc))
    return round(shift_micro_f1*100,2), round(shift_macro_f1*100,2), round(shift_weighted_f1*100,2), round(shift_acc*100,2)

def metricsShiftEmotion(labels, preds, speakers, dataset):

    if dataset == "IEMOCAP":
        label_sum = 6
    elif dataset == "MELD":
        label_sum = 7
    bias = 0
    speaker_values = {}
    for speaker in speakers:
        cur_label = labels[bias:bias+len(speaker)]
        cur_pred = preds[bias:bias+len(speaker)]
        bias = bias + len(speaker)
        speaker = np.array(speaker)
        speaker_set = set(speaker)
        for s in speaker_set:
            if s not in speaker_values.keys():
                speaker_values[s] = []
            indices = np.where(speaker == s)[0]
            speaker_values[s].append((cur_label[indices], cur_pred[indices]))
    w_shift_erc_labels = []
    w_shift_erc_preds = []
    wo_shift_erc_labels = []
    wo_shift_erc_preds = []

    for s in speaker_values.keys():
        value = speaker_values[s]
        value_shift_label = []
        value_shift_pred = []
        for item in value:
            shift_info1 = np.zeros_like(item[0])
            shift_info2 = np.zeros_like(item[0])
            shift_info1[:-1] = (item[0][:-1] == item[0][1:])
            shift_info2[1:] = (item[0][:-1] == item[0][1:])
            shift_info = shift_info1 + shift_info2
            w_shift_erc_labels.append(item[0][shift_info == 1])
            w_shift_erc_preds.append(item[1][shift_info == 1])
            wo_shift_erc_labels.append(item[0][shift_info == 0])
            wo_shift_erc_preds.append(item[1][shift_info == 0])
            
    w_shift_erc_labels = np.concatenate(w_shift_erc_labels)
    w_shift_erc_preds = np.concatenate(w_shift_erc_preds)
    wo_shift_erc_labels = np.concatenate(wo_shift_erc_labels)
    wo_shift_erc_preds = np.concatenate(wo_shift_erc_preds)

    print("with shift erc:")
    print(metrics.classification_report(w_shift_erc_labels, w_shift_erc_preds, digits=4))
    print("micro f1 = ", metrics.f1_score(w_shift_erc_labels, w_shift_erc_preds, average='micro'))
    print("macro f1 = ", metrics.f1_score(w_shift_erc_labels, w_shift_erc_preds, average='macro'))
    print("weighted f1 = ", metrics.f1_score(w_shift_erc_labels, w_shift_erc_preds, average='weighted'))
    print("acc = ", metrics.accuracy_score(w_shift_erc_labels, w_shift_erc_preds))

    print("without shift erc:")
    print(metrics.classification_report(wo_shift_erc_labels, wo_shift_erc_preds, digits=4))
    print("micro f1 = ", metrics.f1_score(wo_shift_erc_labels, wo_shift_erc_preds, average='micro'))
    print("macro f1 = ", metrics.f1_score(wo_shift_erc_labels, wo_shift_erc_preds, average='macro'))
    print("weighted f1 = ", metrics.f1_score(wo_shift_erc_labels, wo_shift_erc_preds, average='weighted'))
    print("acc = ", metrics.accuracy_score(wo_shift_erc_labels, wo_shift_erc_preds))


def constructShiftLabel(qmask, lengths, label, class_num):
    shift_label = []
    qmask = torch.cat([qmask[:lengths[i], i ,:] for i in range(len(lengths))], dim=0)
    uttr_count = 0
    for dia_len in lengths:
        dia_speaker = sorted(set(torch.nonzero(qmask[uttr_count: uttr_count+dia_len])[:,1].tolist()))
        for speaker in dia_speaker:
            speaker_index = torch.nonzero(qmask[uttr_count: uttr_count+dia_len])[:,1] == speaker
            current_speaker_label = label[uttr_count: uttr_count+dia_len][speaker_index.bool()]
            if current_speaker_label.shape[0] > 1:
                for i in range(current_speaker_label.shape[0]-1):
                    shift_label.append(current_speaker_label[i] * class_num + current_speaker_label[i+1])
        uttr_count = uttr_count + dia_len

    return torch.tensor(shift_label).to(qmask.device)




def seed_everything(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False, use_multiemo=False):
    trainset = MELDDataset(data_path, use_multiemo=use_multiemo)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(data_path, train=False, use_multiemo=use_multiemo)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False, use_multiemo=False):
    trainset = IEMOCAPDataset(path=data_path, use_multiemo=use_multiemo)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=data_path, train=False, use_multiemo=use_multiemo)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_graph_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, scheduler=None, cuda_flag=False, modals=None, target_names=None,
                              test_label=False, tensorboard=False):
    losses, preds, labels = [], [], []
    shift_labels, shift_preds = [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda_flag: ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train_flag or optimizer != None
    if train_flag:
        model.train()
    else:
        model.eval()
    speaker_all = []
    seed_everything(seed=args.seed)
    for data in dataloader:
        if train_flag:
            optimizer.zero_grad()

        if args.dataset == 'IEMOCAP':
            textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label, persona = [d.cuda() for d in data[:-2]] if cuda_flag else data[:-2]
            speaker = data[-2]
        elif args.dataset == 'MELD':
            textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label, persona = [d.cuda() for d in data[:-3]] if cuda_flag else data[:-3]
            speaker = data[-3]
            persona_info = data[-2]
            persona_job = torch.cat([item[0] for item in persona_info], dim=0)
            persona_sex = torch.cat([item[1] for item in persona_info], dim=0)
            persona_personality = torch.cat([item[2] for item in persona_info], dim=0)
            persona_info = torch.cat([persona_job, persona_sex, persona_personality, torch.mean(torch.cat([persona_job.unsqueeze(0), persona_sex.unsqueeze(0), persona_personality.unsqueeze(0)], dim=0), dim=0)], dim=-1).to(textf1.device)
            
        speaker_all.extend(speaker)
        lengths = [int(torch.sum(umask[:,i])) for i in range(umask.shape[1])]
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        label_shift = constructShiftLabel(qmask, lengths, label, 6 if args.dataset=='IEMOCAP' else 7)
        log_prob, log_prob_shift, hidden, hidden_shift = model([textf1, textf2, textf3, textf4], qmask, umask, lengths, acouf, visuf, speaker=speaker, persona=persona)
        loss_erc = loss_f['erc'](input=log_prob, target=label)
        if args.wo_shiftcl:
            loss_shift = loss_f['shift'](log_prob_shift, label_shift)
            loss = loss_erc * (1/float(loss_erc) if float(loss_erc) != 0 else 0) + \
                loss_shift * (1/float(loss_shift) if float(loss_shift) != 0 else 0)
        else:
            loss_shift = loss_f['shift'](log_prob_shift, label_shift)
            shift_features_pair, shift_features_pair_label = shiftContrastLearning(shift_features=hidden_shift, shift_labels=label_shift, sample_num=args.sample_num, dataset=args.dataset)
            loss_shift_cl = loss_f['shift_cl'](features=shift_features_pair, labels=shift_features_pair_label, weight=create_class_weight_SCL(shift_features_pair_label))
            loss = loss_erc * (1/float(loss_erc) if float(loss_erc) != 0 else 0) + \
                loss_shift * (1/float(loss_shift) if float(loss_shift) != 0 else 0) + \
                loss_shift_cl * (1/float(loss_shift_cl) if float(loss_shift_cl) !=0 else 0)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        shift_preds.append(torch.argmax(log_prob_shift, 1).cpu().numpy())
        shift_labels.append(label_shift.cpu().numpy())
        if train_flag:
            loss.backward()
            if tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        if args.wo_crosstask:
            losses.append(float(loss_erc)+float(loss_shift))
        elif args.wo_shiftcl:
            losses.append(float(loss_erc)+float(loss_shift))
        else:
            losses.append(float(loss_erc)+float(loss_shift)+float(loss_shift_cl))

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        shift_preds = np.concatenate(shift_preds)
        shift_labels = np.concatenate(shift_labels)
        if not train_flag :
            print("shift metrics: ACC = {}, F1 score = {}".format(accuracy_score(shift_labels, shift_preds), f1_score(shift_labels, shift_preds, average='weighted')))
    else:
        return [], [], float('nan'), float('nan'), [], [], float('nan'), []
    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)
    print(metrics.confusion_matrix(labels, preds))
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_each = metrics.classification_report(labels, preds, target_names=target_names, digits=4)
    all_acc = ["ACC"]
    for i in range(len(target_names)):
        all_acc.append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    return all_each, all_acc, avg_loss, avg_accuracy, labels, preds, avg_fscore, [vids, ei, et, en, el]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--data_dir', type=str, default='../data/iemocap/IEMOCAP_features.pkl', help='dataset dir')

    parser.add_argument('--multi_modal', action='store_true', default=True, help='whether to use multimodal information')

    parser.add_argument('--modals', default='avl', help='modals to fusion: avl')

    parser.add_argument('--mm_fusion_mthd', default='concat_subsequently',
                        help='method to use multimodal information: mfn, concat, gated, concat_subsequently,mfn_only,tfn_only,lmf_only')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--base_model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU/None')

    parser.add_argument('--graph_model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--graph_type', default='GDF', help='relation/GCN3/DeepGCN/GF/GF2/GDF')

    parser.add_argument('--graph_construct', default='direct', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--nodal_attention', action='store_true', default=True, help='whether to use nodal attention in graph model')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--use_residue', action='store_true', default=True, help='whether to use residue information or not')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--active_listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--use_crn_speaker', action='store_true', default=True, help='whether to use use crn_speaker embedding')

    parser.add_argument('--speaker_weights', type=str, default='3-0-1', help='speaker weight 0-0-0')

    parser.add_argument('--use_speaker', action='store_true', default=False, help='whether to use speaker embedding')

    parser.add_argument('--reason_flag', action='store_true', default=False, help='reason flag')

    parser.add_argument('--epochs', type=int, default=80, metavar='E', help='number of epochs')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.0, metavar='valid_rate', help='valid rate, 0.0/0.1')

    parser.add_argument('--modal_weight', type=float, default=1.0, help='modal weight 1/0.7')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=16, help='Deep_GCN_nlayers')

    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec_dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.6, metavar='dropout', help='dropout rate')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha 0.1/0.2')

    parser.add_argument('--lamda', type=float, default=0.5, help='eta 0.5/0')

    parser.add_argument('--gamma', type=float, default=0.5, help='gamma 0.5/1/2')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--loss', default="FocalLoss", help='loss function: FocalLoss/NLLLoss')

    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--save_model_dir', type=str, default='../outputs/iemocap_demo/', help='saved model dir')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--test_label', action='store_true', default=False, help='whether do test only')

    parser.add_argument('--load_model', type=str, default='../outputs/iemocap_demo/model_4.pkl', help='trained model dir')

    parser.add_argument('--seed', type=int, default=3407, help='random seed')

    parser.add_argument('--patience', type=int, default=50, help='early stop')
    
    parser.add_argument('--mtl', default=False, action='store_true', help='whether to use multi-task learning')

    parser.add_argument('--use_clone', default=False, action='store_true', help='multi-task learning with clone')
    
    parser.add_argument('--gat', default=False, action='store_true', help='whether to use gat')

    parser.add_argument('--gcn', default=False, action='store_true', help='whether to use gcn')

    parser.add_argument('--use_scheduler', default=False, action='store_true')

    parser.add_argument('--hidden_l', type=int, default=300, help='Hidden layer size for language modality')
    parser.add_argument('--hidden_a', type=int, default=300, help='Hidden layer size for acoustic modality')
    parser.add_argument('--hidden_v', type=int, default=300, help='Hidden layer size for visual modality')
    
    parser.add_argument('--persona_l_heads', type=int, default=4, help='Number of heads for language persona')
    parser.add_argument('--persona_a_heads', type=int, default=4, help='Number of heads for acoustic persona')
    parser.add_argument('--persona_v_heads', type=int, default=4, help='Number of heads for visual persona')
    
    parser.add_argument('--persona_l_layer', type=int, default=1, help='Number of layers for language persona')
    parser.add_argument('--persona_a_layer', type=int, default=1, help='Number of layers for acoustic persona')
    parser.add_argument('--persona_v_layer', type=int, default=1, help='Number of layers for visual persona')
    
    parser.add_argument('--interactive_layer', type=int, default=2, help='Number of interactive layers')
    parser.add_argument('--interactive_heads', type=int, default=4, help='Number of interactive heads')
    
    parser.add_argument('--persona_transform', action='store_true', help='Enable persona transform')
    
    parser.add_argument('--dropout_forward', type=float, default=0.5, help='Dropout rate for forward pass')
    parser.add_argument('--dropout_persona_lstm_modeling', type=float, default=0.5, help='Dropout rate for persona LSTM modeling')
    parser.add_argument('--dropout_interactive', type=float, default=0.1, help='Dropout rate for interactive layers')
    parser.add_argument('--dropout_persona', type=float, default=0.1, help='Dropout rate for persona layers')
    parser.add_argument('--dropout_smax_erc', type=float, default=0.5, help='Dropout rate for softmax in ERC')
    parser.add_argument('--dropout_smax_shift', type=float, default=0.5, help='Dropout rate for softmax in shift')
    
    parser.add_argument('--erc_windows', type=int, default=1, help='Window size for ERC')
    parser.add_argument('--shift_windows', type=int, default=1, help='Window size for shift')
    parser.add_argument('--interactive_windows', type=int, default=1, help='Window size for interactive layer')
    parser.add_argument('--use_multiemo', default=False, action='store_true')
    parser.add_argument('--sample_num', default=50, type=int)
    parser.add_argument('--wo_persona', default=False, action='store_true')
    parser.add_argument('--wo_crosstask', default=False, action='store_true')
    parser.add_argument('--wo_shiftcl', default=False, action='store_true')
    parser.add_argument('--index', default=1,type=int)
    args = parser.parse_args()

    today = datetime.datetime.now()
    print(args)
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + 'using_lstm_' + args.dataset
    else:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + str(args.Deep_GCN_nlayers) + '_' + args.dataset

    if args.use_speaker:
        name_ = name_ + '_speaker'
    if args.use_modal:
        name_ = name_ + '_modal'

    cuda_flag = torch.cuda.is_available() and not args.no_cuda

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    # D_text = feat2dim['textCNN'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_text']
    # if args.mtl:
    D_text = 1024
    D_m = D_text
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    n_speakers, n_classes, class_weights, target_names = -1, -1, None, None
    if args.dataset == 'IEMOCAP':
        n_speakers, n_classes = 2, 6
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        class_weights = torch.FloatTensor([1 / 0.086747,
                                           1 / 0.144406,
                                           1 / 0.227883,
                                           1 / 0.160585,
                                           1 / 0.127711,
                                           1 / 0.252668])
    if args.dataset == 'MELD':
        n_speakers, n_classes = 9, 7

        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
        class_weights = torch.FloatTensor([1.0 / 0.466750766,
                                           1.0 / 0.122094071,
                                           1.0 / 0.027752748,
                                           1.0 / 0.071544422,
                                           1.0 / 0.171742656,
                                           1.0 / 0.026401153,
                                           1.0 / 0.113714183])

    seed_everything(seed=args.seed)
    model = GATModel(base_model=args.base_model, D_m=D_m, D_m_v=D_visual, D_m_a=D_audio, n_speakers=n_speakers,
                graph_type='relation', modals=args.modals, dataset=args.dataset, speaker_weights='1-1-1',
                hidden_l=args.hidden_l, hidden_a=args.hidden_a, hidden_v=args.hidden_v,
                persona_l_heads=args.persona_l_heads, persona_a_heads=args.persona_a_heads, persona_v_heads=args.persona_v_heads,
                persona_l_layer=args.persona_l_layer, persona_a_layer=args.persona_a_layer, persona_v_layer=args.persona_v_layer,
                interactive_layer=args.interactive_layer, interactive_heads=args.interactive_heads,
                persona_transform=args.persona_transform,
                dropout_forward=args.dropout_forward, dropout_persona_lstm_modeling=args.dropout_persona_lstm_modeling, 
                dropout_interactive=args.dropout_interactive, dropout_persona=args.dropout_persona,
                dropout_smax_erc=args.dropout_smax_erc, dropout_smax_shift=args.dropout_smax_shift,
                erc_windows=args.erc_windows, shift_windows=args.shift_windows, interactive_windows=args.interactive_windows,
                av_using_lstm=args.av_using_lstm, norm='', wo_persona=args.wo_persona, wo_crosstask=args.wo_crosstask
        )
    print('{} with {} as base model'.format("PCGNet", args.base_model))

    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(modals))

    if cuda_flag:
        # torch.cuda.set_device(0)
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')


    loss_f = {}
    if args.loss == 'FocalLoss' and args.graph_model:
        # FocalLoss
        loss_f['erc'] = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
        print("focal-loss")
        print("args.gamma = ", args.gamma)
        print("args.alpha = ", class_weights)
    else:
        # NLLLoss
        loss_f['erc'] = nn.NLLLoss(class_weights if args.class_weight else None)
        print("nll-loss")
    loss_f['shift'] = nn.CrossEntropyLoss()
    loss_f['shift_cl'] = CKSCL()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)




    # warm_steps = 0.1
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.95, patience = 5, threshold = 1e-6, verbose = True)
        args.valid_rate = 0.05
    else:
        scheduler = None
    if args.dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(data_path=args.data_dir,
                                                                   valid_rate=args.valid_rate,
                                                                   batch_size=batch_size,
                                                                   num_workers=0,
                                                                      use_multiemo=args.use_multiemo)
    elif args.dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(data_path=args.data_dir,
                                                                      valid_rate=args.valid_rate,
                                                                      batch_size=batch_size,
                                                                      num_workers=0,
                                                                      use_multiemo=args.use_multiemo)
    else:
        train_loader, valid_loader, test_loader = None, None, None
        print("There is no such dataset")
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    if args.test_label and args.graph_model:
        model = torch.load(args.load_model)
        all_each, all_acc, test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(model=model,
                                                                                                                  loss_f=loss_f,
                                                                                                                  dataloader=test_loader,
                                                                                                                  train_flag=False,
                                                                                                                  cuda_flag=cuda_flag,
                                                                                                                  modals=args.modals,
                                                                                                                  target_names=target_names,
                                                                                                                  test_label=True)
        print('# test_label,test_pred', len(test_label), len(test_pred))
        # import numpy as np

        print(all_each)
        print(all_acc)
        exit(0)

    all_test_fscore, all_test_acc = [], []
    best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
    patience2 = 0
    for e in range(n_epochs):
        start_time = time.time()
        _, _, train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_graph_model(model=model,
                                                                                        loss_f=loss_f,
                                                                                        dataloader=train_loader,
                                                                                        epoch=e,
                                                                                        train_flag=True,
                                                                                        optimizer=optimizer,
                                                                                        scheduler=scheduler,
                                                                                        cuda_flag=cuda_flag,
                                                                                        modals=args.modals,
                                                                                        target_names=target_names)
        _, _, valid_loss, valid_acc, _, _, valid_fscore, _ = train_or_eval_graph_model(model=model,
                                                                                        loss_f=loss_f,
                                                                                        dataloader=valid_loader,
                                                                                        epoch=e,
                                                                                        scheduler=scheduler,
                                                                                        cuda_flag=cuda_flag,
                                                                                        modals=args.modals,
                                                                                        target_names=target_names)
        all_each, all_acc, test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(model=model,
                                                                                                                    loss_f=loss_f,
                                                                                                                    dataloader=test_loader,
                                                                                                                    epoch=e,
                                                                                                                    scheduler=scheduler,
                                                                                                                    cuda_flag=cuda_flag,
                                                                                                                    modals=args.modals,
                                                                                                                    target_names=target_names)
        if scheduler != None:
            scheduler.step(valid_loss)
            optimizer.zero_grad()


        all_test_fscore.append(test_fscore)
        all_test_acc.append(test_acc)
        if args.valid_rate > 0:
            eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
        else:
            eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
        if e == 0 or best_eval_fscore < eval_fscore:
            patience = 0
            best_epoch, best_eval_fscore = e, eval_fscore
        else:
            patience += 1
        if best_eval_loss is None:
            best_eval_loss = eval_loss
            best_epoch2 = 0
        else:
            if eval_loss < best_eval_loss:
                best_epoch2, best_eval_loss = e, eval_loss
                patience2 = 0
            else:
                patience2 += 1

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                       round(time.time() - start_time, 2)))

        print(all_each)
        print(all_acc)

        if patience >= args.patience and patience2 >= args.patience:
            print('Early stoping...', patience, patience2)
            break

    print('Final Test performance...')
    print('Early stoping...', patience, patience2)
    print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch, best_eval_fscore,
                                                                                                all_test_acc[best_epoch] if best_epoch >= 0 else 0,
                                                                                                all_test_fscore[best_epoch] if best_epoch >= 0 else 0))
