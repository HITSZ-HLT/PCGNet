import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import MessagePassing
from torch.autograd import Variable
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from loss import *







class personaGATConvLayer(MessagePassing):
    def __init__(self, in_channels, persona_channels, out_channels, 
                 heads=1, 
                 concat=True, 
                 bias=True,
                 negative_slope= 0.2,
                 dropout=0.6,
                 training=True):
        super(personaGATConvLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.bias = bias
        self.negative_slope = negative_slope
        self.training = training
        self.persona_gate = nn.ModuleList([Linear(in_channels, 1, weight_initializer='glorot') for _ in range(heads)])
        self.lins_persona = nn.ModuleList([Linear(persona_channels, int(out_channels/heads), bias=False, weight_initializer='glorot') for _ in range(heads)])
        self.lins = nn.ModuleList([Linear(in_channels, int(out_channels/heads), bias=False, weight_initializer='glorot') for _ in range(heads)])
        self.attentions = nn.ModuleList([Linear(int(out_channels/heads)*2, 1, bias=False, weight_initializer='glorot') for _ in range(heads)])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()    

    def reset_parameters(self):
        super().reset_parameters()
        for head in range(self.heads):
            self.persona_gate[head].reset_parameters()
            self.lins_persona[head].reset_parameters()
            self.lins[head].reset_parameters()
            self.attentions[head].reset_parameters()
            zeros(self.bias)


    def forward(self, x, persona,edge_index, size=None):
        # x = F.dropout(x, p=0.6, training=self.training)
        persona_features = []
        x_features = []

        for head in range(self.heads):
            persona_features.append(self.persona_gate[head](x) * self.lins_persona[head](persona))
            x_features.append(self.lins[head](x))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        scores = []
        for head in range(self.heads):
            persona_i = persona_features[head][edge_index[0]]
            persona_j = persona_features[head][edge_index[1]]

            score = self.attentions[head](torch.cat([persona_i, persona_j], dim=-1))
            score = F.leaky_relu(score, negative_slope=self.negative_slope)
            score = softmax(score, edge_index[1], num_nodes=size)
            scores.append(score)


        out_features = []
        for head in range(self.heads):
            out = self.propagate(edge_index, x=x_features[head], norm=scores[head])
            out_features.append(out)

        out = torch.cat(out_features, dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out






class personaGAT(nn.Module):
    def __init__(self, in_channels, persona_channels, out_channels, heads, layer_nums, dropout, training=True):
        super(personaGAT, self).__init__()
        self.layer_nums = layer_nums
        self.dropout = dropout
        self.training = training
        self.GATConv = nn.ModuleList([personaGATConvLayer(in_channels=in_channels, persona_channels=persona_channels, out_channels=out_channels, heads=heads)] \
                                     + [personaGATConvLayer(in_channels=out_channels, persona_channels=persona_channels, out_channels=out_channels, heads=heads) for _ in range(self.layer_nums-1)])

    def forward(self, x, persona, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.layer_nums-1):
            out = F.dropout(self.GATConv[i](x, persona, edge_index), p=self.dropout, training=self.training)
            out = F.elu(out)
            x = x + out
            # x = F.elu(self.GATConv[i](x, persona, edge_index))
            # x = F.dropout(x, p=self.dropout, training=self.training)
        out = F.dropout(self.GATConv[-1](x, persona, edge_index), p=self.dropout, training=self.training)
        out = F.elu(out)
        x = x + out
        return x












class interactiveGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, layer_nums, dropout=0.6, training=True):
        super(interactiveGAT, self).__init__()
        self.layer_nums = layer_nums
        self.training = training
        self.dropout = dropout
        self.GATConv = nn.ModuleList([RealtionalGATConv(in_channels, out_channels, heads) for _ in range(self.layer_nums-1)] + \
                                     [RealtionalGATConv(out_channels, out_channels, heads)])

    def forward(self, x, edge_index, edge_indice, edge_type, edge_dialog):

        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.layer_nums-1):
            out = F.dropout(self.GATConv[i](x, edge_index, edge_indice, edge_type, edge_dialog), p=self.dropout, training=self.training)
            out = F.elu(out)
            x = x + out

            # x = F.elu(self.GATConv[i](x, edge_index, edge_type, edge_dialog))
            # x = F.dropout(x, p=self.dropout, training=self.training)
        out = F.dropout(self.GATConv[-1](x, edge_index, edge_indice, edge_type, edge_dialog), p=self.dropout, training=self.training)
        out = F.elu(out)
        x = x + out

        # x = self.GATConv[-1](x, edge_index, edge_type, edge_dialog)
        return x



class RealtionalGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, relation_nums=8,
                 concat=True, bias=True, negative_slope= 0.2, dropout=0.6, training=True):
        super(RealtionalGATConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.bias = bias
        self.negative_slope = negative_slope
        self.training = training
        self.position_encodings_weight = Parameter(torch.Tensor(relation_nums, heads)) 
        self.position_encodings_bias = Parameter(torch.Tensor(relation_nums, heads)) 
        self.lins = nn.ModuleList([Linear(in_channels, int(out_channels/heads), bias=False, weight_initializer='glorot') for _ in range(heads)])
        self.attentions = nn.ModuleList([Linear(int(out_channels/heads)*2, 1, bias=False, weight_initializer='glorot') for _ in range(heads)])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()    
        


    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.position_encodings_weight)
        glorot(self.position_encodings_bias)
        for head in range(self.heads):
            self.lins[head].reset_parameters()
            self.attentions[head].reset_parameters()
            zeros(self.bias)

    def normalize_tensor_by_labels(self, tensor, labels):
        unique_labels = labels.unique()
        normalized_tensor = torch.zeros_like(tensor)

        for label in unique_labels:
            indices = (labels == label).nonzero(as_tuple=True)[0]
            label_data = tensor[indices]
            normalized_label_data = (label_data - label_data.mean()) / label_data.std()
            # 将归一化的数据放回原tensor
            normalized_tensor[indices] = normalized_label_data

        return normalized_tensor



    def normalize_tensor_by_labels(self, tensor, edge_type, edge_dialog):
        unique_types = edge_type.unique()
        normalized_tensor = torch.zeros_like(tensor).float()

        for dia_index in range(int(torch.max(edge_dialog))+1):
            dia_uttr_indices = torch.where(edge_dialog == dia_index)[0]

            for type in unique_types:
                type_indices = (edge_type == type).nonzero(as_tuple=True)[0]


                mask = torch.isin(dia_uttr_indices, type_indices)
                intersection_indices = dia_uttr_indices[mask]
                if intersection_indices.shape[0] != 0:
                    type_data = tensor[intersection_indices].float()
                    # normalized_type_data = type_data / torch.norm(type_data, p=1)

                    normalized_type_data = type_data / torch.max(torch.abs(type_data))



                    # if torch.sum(torch.isnan(normalized_type_data)) != 0:
                    #     pass
                    normalized_tensor[intersection_indices] = normalized_type_data

        return normalized_tensor





    def forward(self, x, edge_index, edge_indice, edge_type, edge_dialog, size=None):

        x = F.dropout(x, p=self.dropout, training=self.training)

        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out_features = []

        for head in range(self.heads):
            # relative_index = self.normalize_tensor_by_labels((edge_index[0] - edge_index[1]), edge_type, edge_dialog)
            # position_encoding = self.position_encodings_weight[edge_type,head]*(edge_indice[0] - edge_indice[1]) + self.position_encodings_bias[edge_type,head]
            x_features = self.lins[head](x)

            # score = self.attentions[head](torch.cat([x_features[edge_index[0]], x_features[edge_index[1]]], dim=-1)) + position_encoding.unsqueeze(1)
            score = self.attentions[head](torch.cat([x_features[edge_index[0]], x_features[edge_index[1]]], dim=-1))
            score = F.leaky_relu(score, negative_slope=self.negative_slope)
            score = softmax(score, edge_index[1], num_nodes=size)
            out = self.propagate(edge_index, x=x_features, norm=score)
            out_features.append(out)

        out = torch.cat(out_features, dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out











class GATModel(nn.Module):

    def __init__(self, base_model, D_m, D_m_v, D_m_a, n_speakers,
                 graph_type='relation',
                 modals='avl', dataset='IEMOCAP', 
                 speaker_weights='1-1-1',
                 hidden_l=300, hidden_a=300, hidden_v=300,
                 persona_l_heads=4, persona_a_heads=4, persona_v_heads=4,
                 persona_l_layer=1, persona_a_layer=1, persona_v_layer=1,
                 interactive_layer=2, interactive_heads=4, persona_transform=False,
                 dropout_forward=0.5, dropout_persona_lstm_modeling=0.5, dropout_interactive=0.1, dropout_persona=0.1,
                 dropout_smax_erc=0.5, dropout_smax_shift=0.5,
                 erc_windows=1, shift_windows=1, interactive_windows=1,
                 av_using_lstm=False,
                 norm='BN',
                 wo_persona=False, wo_crosstask=False
                 ):

        super(GATModel, self).__init__()

        self.modals = [x for x in modals]  # [a, v, l]
        self.n_speakers = n_speakers
        self.speaker_weights = list(map(float, speaker_weights.split('-')))

        
        self.class_num = 6 if dataset == 'IEMOCAP' else 7
        
        self.use_bert_seq = False
        self.dataset = dataset
        self.wo_persona = wo_persona
        self.wo_crosstask = wo_crosstask

        hidden_l = hidden_l
        hidden_a = hidden_a
        hidden_v = hidden_v

        self.dropout_forward = nn.Dropout(dropout_forward)

        #################
        # norm stragy
        self.norm_strategy = norm
        if self.norm_strategy == 'BN':
            self.normBNa = nn.BatchNorm1d(D_m, affine=True)
            self.normBNb = nn.BatchNorm1d(D_m, affine=True)
            self.normBNc = nn.BatchNorm1d(D_m, affine=True)
            self.normBNd = nn.BatchNorm1d(D_m, affine=True)
        elif self.norm_strategy == 'LN':
            self.normLNa = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNb = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNc = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNd = nn.LayerNorm(D_m, elementwise_affine=True)

        ##################
        # persona modeling
        self.linear_l = nn.Linear(D_m, hidden_l)
        self.linear_a = nn.Linear(D_m_a, hidden_a)
        self.linear_v = nn.Linear(D_m_v, hidden_v)

        self.lstm_l = nn.GRU(input_size=hidden_l, hidden_size=int(hidden_l/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)
        self.speaker_weights_l1 = nn.Linear(hidden_l, hidden_l)
        self.speaker_weights_l2 = nn.Linear(hidden_l, hidden_l)
        self.speaker_weights_a1 = nn.Linear(hidden_a, hidden_a)
        self.speaker_weights_a2 = nn.Linear(hidden_a, hidden_a)
        self.speaker_weights_v1 = nn.Linear(hidden_v, hidden_v)
        self.speaker_weights_v2 = nn.Linear(hidden_v, hidden_v)
        self.av_using_lstm = av_using_lstm
        if self.av_using_lstm:
            self.lstm_a = nn.GRU(input_size=hidden_a, hidden_size=int(hidden_a/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)
            self.lstm_v = nn.GRU(input_size=hidden_v, hidden_size=int(hidden_v/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)

        self.rnn_parties_l = nn.GRU(input_size=hidden_l, hidden_size=int(hidden_l/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)
        self.rnn_parties_a = nn.GRU(input_size=hidden_a, hidden_size=int(hidden_a/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)
        self.rnn_parties_v = nn.GRU(input_size=hidden_v, hidden_size=int(hidden_v/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)


        ##################
        # persona inject
        if self.wo_persona == False:
            self.persona_hidden = 5
            self.persona_transform = persona_transform
            if persona_transform:
                self.persona_hidden = hidden_l
                self.linear_persona = nn.Linear(5, self.persona_hidden)
            
            if 'l' in self.modals:
                self.personaGAT_l = personaGAT(in_channels=hidden_l, persona_channels=self.persona_hidden, out_channels=hidden_l, heads=persona_l_heads, layer_nums=persona_l_layer, dropout=dropout_persona)
            if 'a' in self.modals:
                self.personaGAT_a = personaGAT(in_channels=hidden_a, persona_channels=self.persona_hidden, out_channels=hidden_a, heads=persona_a_heads, layer_nums=persona_a_layer, dropout=dropout_persona)
            if 'v' in self.modals:
                self.personaGAT_v = personaGAT(in_channels=hidden_v, persona_channels=self.persona_hidden, out_channels=hidden_v, heads=persona_v_heads, layer_nums=persona_v_layer, dropout=dropout_persona)
            self.persona_transformation_l1 = nn.Linear(hidden_l, hidden_l)
            self.persona_transformation_l2 = nn.Linear(hidden_l, hidden_l)
            self.persona_transformation_a1 = nn.Linear(hidden_a, hidden_a)
            self.persona_transformation_a2 = nn.Linear(hidden_a, hidden_a)
            self.persona_transformation_v1 = nn.Linear(hidden_v, hidden_v)
            self.persona_transformation_v2 = nn.Linear(hidden_v, hidden_v)
        

        ###################
        # cross task interactive
        self.erc_windows = erc_windows                   ### min erc_windows = 1
        self.shift_windows = shift_windows               ### min shift_windows = 1
        self.interactive_windows = interactive_windows   ### min interactive_windows = 0
        self.interactiveGAT = interactiveGAT(in_channels=hidden_l, out_channels=hidden_l, heads=interactive_heads, layer_nums=interactive_layer, dropout=dropout_interactive)
        self.interactive_transformation1 = nn.Linear(hidden_l, hidden_l)
        self.interactive_transformation2 = nn.Linear(hidden_l, hidden_l)

        ###################
        # smax
        self.smax_fc = nn.Linear(len(modals)*hidden_l, self.class_num)
        self.dropout_smax_erc = nn.Dropout(p=dropout_smax_erc)
        self.smax_fc_shift = nn.Linear(2*len(modals)*hidden_l, self.class_num**2)
        self.dropout_smax_shift = nn.Dropout(p=dropout_smax_shift)
        self.reset_parameters()



    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_l.weight)
        nn.init.xavier_uniform_(self.linear_a.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.speaker_weights_l1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_a1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_v1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_l2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_a2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_v2.weight)
        # TODO
        if self.wo_persona == False:
            nn.init.xavier_uniform_(self.persona_transformation_l1.weight)
            nn.init.xavier_uniform_(self.persona_transformation_a1.weight)
            nn.init.xavier_uniform_(self.persona_transformation_v1.weight)
            nn.init.xavier_uniform_(self.persona_transformation_l2.weight)
            nn.init.xavier_uniform_(self.persona_transformation_a2.weight)
            nn.init.xavier_uniform_(self.persona_transformation_v2.weight)
        nn.init.xavier_uniform_(self.interactive_transformation1.weight)
        nn.init.xavier_uniform_(self.interactive_transformation2.weight)
        nn.init.xavier_uniform_(self.smax_fc.weight)
        nn.init.xavier_uniform_(self.smax_fc_shift.weight)


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()
        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        return pad_sequence(xfs)


    def constructPersonaEdge(self, qmask, lengths):
        persona_edge_nums = 200000
        edge_index = torch.zeros([2, persona_edge_nums], dtype=torch.long)
        edge_count = 0
        uttr_bias = 0
        for dia_len in lengths:
            cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]
            cur_dia_speakers = sorted(list(set(torch.nonzero(cur_dia_qmask)[:, 1].tolist())))
            cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]
            speaker_past_say = {cur_dia_speaker:0 for cur_dia_speaker in cur_dia_speakers}
            for i in range(dia_len):
                uttr_speaker = int(cur_dia_uttr_speaker[i])
                for j in range(speaker_past_say[uttr_speaker], i):
                    edge_index[0][edge_count] = uttr_bias + j
                    edge_index[1][edge_count] = uttr_bias + i
                    edge_count += 1
                speaker_past_say[uttr_speaker] = i
            uttr_bias += dia_len
        edge_index = edge_index[:,:edge_count]
        return edge_index.to(qmask.device)


    def constructErcEdges(self, text, audio, visul, qmask, lengths, erc_windows=1):
        erc_edge_nums = 200000
        modals = []
        modal_data = {}
        modal_bias = {}
        if text != None:
            modals.append('l')
            modal_data['l'] = text
            modal_bias['l'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if audio != None:
            modals.append('a')
            modal_data['a'] = audio
            modal_bias['a'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if visul != None:
            modals.append('v')
            modal_data['v'] = visul
            modal_bias['v'] = (len(modal_data.keys()) - 1) * sum(lengths)
        
        erc_features = torch.cat([modal_data[modal] for modal in modals], dim=0)
        erc_edge_index = torch.zeros([2, erc_edge_nums], dtype=torch.long)
        erc_edge_indice = torch.zeros([2, erc_edge_nums], dtype=torch.long)
        erc_edge_type = torch.zeros([erc_edge_nums], dtype=torch.long)
        erc_edge_dialog = torch.zeros([erc_edge_nums], dtype=torch.long)
        erc_edge_count = 0

        #################################
        # add utterance multimodal edges
        # edges count: (len(modals) * len(modals) - len(modals)) * uttr_count
        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            for uttr_index in range(dia_len):
                for modal1 in modals:
                    for modal2 in modals:
                        erc_edge_index[0][erc_edge_count] = modal_bias[modal1] + uttr_bias + uttr_index
                        erc_edge_index[1][erc_edge_count] = modal_bias[modal2] + uttr_bias + uttr_index
                        erc_edge_indice[0][erc_edge_count] = uttr_index
                        erc_edge_indice[1][erc_edge_count] = uttr_index
                        erc_edge_dialog[erc_edge_count] = dia_index
                        erc_edge_type[erc_edge_count] = 0
                        erc_edge_count = erc_edge_count + 1
            uttr_bias += dia_len
                
        ##################################
        # add neighbours edges
        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]
            cur_dia_speakers = sorted(list(set(torch.nonzero(cur_dia_qmask)[:, 1].tolist())))
            cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]
            speaker_past_say = {cur_dia_speaker:[0] for cur_dia_speaker in cur_dia_speakers}
            for i in range(dia_len):
                uttr_speaker = int(cur_dia_uttr_speaker[i])
                #################################
                # process intra and inter dependency
                for j in range(speaker_past_say[uttr_speaker][-erc_windows] if len(speaker_past_say[uttr_speaker]) >= erc_windows else speaker_past_say[uttr_speaker][0], i):
                    if cur_dia_uttr_speaker[j] == cur_dia_uttr_speaker[i]:
                        for modal in modals:
                            erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                            erc_edge_index[1][erc_edge_count] = uttr_bias + i + modal_bias[modal]
                            erc_edge_indice[0][erc_edge_count] = j
                            erc_edge_indice[1][erc_edge_count] = i
                            # type = 1 mean intra edge
                            erc_edge_type[erc_edge_count] = 1
                            erc_edge_dialog[erc_edge_count] = dia_index
                            erc_edge_count += 1
                    else:
                        for modal in modals:
                            erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                            erc_edge_index[1][erc_edge_count] = uttr_bias + i + modal_bias[modal]
                            erc_edge_indice[0][erc_edge_count] = j
                            erc_edge_indice[1][erc_edge_count] = i
                            # type = 2 mean inter edge
                            erc_edge_type[erc_edge_count] = 2
                            erc_edge_dialog[erc_edge_count] = dia_index
                            erc_edge_count += 1
                speaker_past_say[uttr_speaker].append(i)
            uttr_bias += dia_len
                        
        # assert erc_edge_count == erc_edge_amount
        erc_edge_index = erc_edge_index[:,:erc_edge_count]
        erc_edge_indice = erc_edge_indice[:,:erc_edge_count]
        erc_edge_type = erc_edge_type[:erc_edge_count]
        erc_edge_dialog = erc_edge_dialog[:erc_edge_count]
        return erc_features, erc_edge_index.to(qmask.device), erc_edge_indice.to(qmask.device), erc_edge_type.to(qmask.device), erc_edge_dialog.to(qmask.device), modals, modal_bias
    

    def constructShiftEdges(self, text, audio, visul, qmask, lengths, shift_windows=1, edge_index_bias=0):
        shift_edge_nums = 200000
        modals = []
        modal_data = {}
        modal_bias = {}
        if text != None:
            modals.append('l')
            modal_data['l'] = text
            modal_bias['l'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if audio != None:
            modals.append('a')
            modal_data['a'] = audio
            modal_bias['a'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if visul != None:
            modals.append('v')
            modal_data['v'] = visul
            modal_bias['v'] = (len(modal_data.keys()) - 1) * sum(lengths)
        
        shift_features = torch.cat([modal_data[modal] for modal in modals], dim=0)
        
        shift_edge_index = torch.zeros([2, shift_edge_nums], dtype=torch.long)
        shift_edge_indice = torch.zeros([2, shift_edge_nums], dtype=torch.long)
        shift_edge_type = torch.zeros([shift_edge_nums], dtype=torch.long)
        shift_edge_dialog = torch.zeros([shift_edge_nums], dtype=torch.long)
        shift_edge_count = 0

        #################################
        # add utterance multimodal edges
        # edges count: (len(modals) * len(modals) - len(modals)) * uttr_count

        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            for uttr_index in range(dia_len):
                for modal1 in modals:
                    for modal2 in modals:
                        shift_edge_index[0][shift_edge_count] = modal_bias[modal1] + uttr_bias + uttr_index + edge_index_bias
                        shift_edge_index[1][shift_edge_count] = modal_bias[modal2] + uttr_bias + uttr_index + edge_index_bias
                        shift_edge_indice[0][shift_edge_count] = uttr_index
                        shift_edge_indice[1][shift_edge_count] = uttr_index
                        shift_edge_type[shift_edge_count] = 3
                        shift_edge_dialog[shift_edge_count] = dia_index
                        shift_edge_count = shift_edge_count + 1
            uttr_bias += dia_len


        #################################
        # add shift dependency edges
        # edges count:
        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]
            cur_dia_speakers = sorted(list(set(torch.nonzero(cur_dia_qmask)[:, 1].tolist())))
            cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]
            speaker_past_say = {cur_dia_speaker:[0] for cur_dia_speaker in cur_dia_speakers}
            for i in range(dia_len):
                uttr_speaker = int(cur_dia_uttr_speaker[i])
                #################################
                # process intra and inter dependency
                for j in range(speaker_past_say[uttr_speaker][-shift_windows] if len(speaker_past_say[uttr_speaker]) > shift_windows else speaker_past_say[uttr_speaker][0], i):
                    if cur_dia_uttr_speaker[j] == cur_dia_uttr_speaker[i]:
                        for modal in modals:
                            shift_edge_index[0][shift_edge_count] = uttr_bias + j + modal_bias[modal] + edge_index_bias
                            shift_edge_index[1][shift_edge_count] = uttr_bias + i + modal_bias[modal] + edge_index_bias
                            shift_edge_indice[0][shift_edge_count] = j
                            shift_edge_indice[1][shift_edge_count] = i
                            # type = 4 mean intra shift edge
                            shift_edge_type[shift_edge_count] = 4
                            shift_edge_dialog[shift_edge_count] = dia_index
                            shift_edge_count += 1
                    else:
                        for modal in modals:
                            shift_edge_index[0][shift_edge_count] = uttr_bias + j + modal_bias[modal] + edge_index_bias
                            shift_edge_index[1][shift_edge_count] = uttr_bias + i + modal_bias[modal] + edge_index_bias
                            shift_edge_indice[0][shift_edge_count] = j
                            shift_edge_indice[1][shift_edge_count] = i
                            # type = 5 mean inter shift edge
                            shift_edge_type[shift_edge_count] = 5
                            shift_edge_dialog[shift_edge_count] = dia_index
                            shift_edge_count += 1
                speaker_past_say[uttr_speaker].append(i)
            uttr_bias += dia_len
        shift_edge_index = shift_edge_index[:,:shift_edge_count]
        shift_edge_indice = shift_edge_indice[:,:shift_edge_count]
        shift_edge_type = shift_edge_type[:shift_edge_count]
        shift_edge_dialog = shift_edge_dialog[:shift_edge_count]
        return shift_features, shift_edge_index.to(qmask.device), shift_edge_indice.to(qmask.device), shift_edge_type.to(qmask.device), shift_edge_dialog.to(qmask.device), modals, modal_bias
    

    def constructInteractiveEdges(self, erc_features, shift_features, qmask, lengths, modals, modal_bias, interactive_windows = 1):
        task_length = sum(lengths) * len(modals)
        edge_num = 300000
        interactive_edge_index = torch.zeros([2, edge_num], dtype=torch.long)
        interactive_edge_indice = torch.zeros([2, edge_num], dtype=torch.long)
        interactive_edge_type = torch.zeros([edge_num], dtype=torch.long)
        interactive_edge_dialog = torch.zeros([edge_num], dtype=torch.long)
        interactive_edge_count = 0
        interactive_features = torch.cat([erc_features, shift_features], dim=0)
        #################################
        # add utterance multi task edges

        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            for uttr_index in range(dia_len):
                for neighbour_index in range(-interactive_windows, 1):
                    if uttr_index + neighbour_index < 0 or uttr_index + neighbour_index >=dia_len:
                        continue
                    else:
                        for modal_interactive in modals:
                            # erc -> shift
                            interactive_edge_index[0][interactive_edge_count] = uttr_bias + uttr_index + neighbour_index + modal_bias[modal_interactive]
                            interactive_edge_index[1][interactive_edge_count] = uttr_bias + uttr_index + modal_bias[modal_interactive] + task_length
                            interactive_edge_indice[0][interactive_edge_count] = uttr_index + neighbour_index
                            interactive_edge_indice[1][interactive_edge_count] = uttr_index
                            interactive_edge_type[interactive_edge_count] = 6
                            interactive_edge_dialog[interactive_edge_count] = dia_index
                            interactive_edge_count += 1

                            # shift -> erc
                            interactive_edge_index[0][interactive_edge_count] = uttr_bias + uttr_index + neighbour_index + modal_bias[modal_interactive] + task_length
                            interactive_edge_index[1][interactive_edge_count] = uttr_bias + uttr_index + modal_bias[modal_interactive]
                            interactive_edge_indice[0][interactive_edge_count] = uttr_index + neighbour_index
                            interactive_edge_indice[1][interactive_edge_count] = uttr_index
                            interactive_edge_type[interactive_edge_count] = 7
                            interactive_edge_dialog[interactive_edge_count] = dia_index
                            interactive_edge_count += 1
            uttr_bias += dia_len
        interactive_edge_index = interactive_edge_index[:,:interactive_edge_count]
        interactive_edge_indice = interactive_edge_indice[:,:interactive_edge_count]
        interactive_edge_type = interactive_edge_type[:interactive_edge_count]
        interactive_edge_dialog = interactive_edge_dialog[:interactive_edge_count]
        return interactive_features, interactive_edge_index.to(erc_features.device), interactive_edge_indice.to(erc_features.device), interactive_edge_type.to(erc_features.device), interactive_edge_dialog.to(erc_features.device)



    def constructShiftData(self, shift_features, qmask, lengths, node_bias, node_modals):
        features = {}
        for modal in node_modals:
            features[modal] = shift_features[node_bias[modal]: node_bias[modal] + sum(lengths)]
        
        shift_data ={modal:[] for modal in node_modals}
        uttr_count = 0
        for dia_len in lengths:
            dia_speaker = sorted(set(torch.nonzero(qmask[uttr_count: uttr_count+dia_len])[:,1].tolist()))
            for speaker in dia_speaker:
                speaker_index = torch.nonzero(qmask[uttr_count: uttr_count+dia_len])[:,1] == speaker
                for modal in node_modals:
                    current_dia_speaker_features = features[modal][uttr_count: uttr_count+dia_len][speaker_index.bool()]
                    if current_dia_speaker_features.shape[0] > 1:
                        shift_data[modal].append(torch.cat([current_dia_speaker_features[:-1], current_dia_speaker_features[1:]], dim=-1))
            uttr_count = uttr_count + dia_len
        
        for modal in node_modals:
            shift_data[modal] = torch.cat(shift_data[modal], dim=0)
        return shift_data



    def forward(self, U, qmask, umask, lengths, U_a=None, U_v=None, test_label=False, speaker=None, labels=None, persona=None):
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()
        if self.norm_strategy == 'LN':
            r1 = self.normLNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normLNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normLNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normLNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        elif self.norm_strategy == 'BN':
            r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        else:
            pass
        U = (r1 + r2 + r3 + r4)/4
 
 
        U = self.dropout_forward(U)
        U_a = self.dropout_forward(U_a) if U_a is not None else None
        U_v = self.dropout_forward(U_v) if U_v is not None else None

        emotions_a, emotions_v, emotions_l = None, None, None
        if 'l' in self.modals:
            U = self.linear_l(U)
            emotions_l, hidden_l = self.lstm_l(U)
            U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_l.shape[-1]).type(U.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_l(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_l = torch.nn.Sigmoid()(self.speaker_weights_l1(emotions_l)) * emotions_l + torch.nn.Sigmoid()(self.speaker_weights_l2(U_p)) * U_p


        if 'a' in self.modals:
            U_a = self.linear_a(U_a)
            emotions_a = U_a
            if self.av_using_lstm:
                emotions_a, hidden_a = self.lstm_a(U_a)
            U_, qmask_ = U_a.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_a.shape[-1]).type(U_a.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_a(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_a = torch.nn.Sigmoid()(self.speaker_weights_a1(emotions_a)) * emotions_a + torch.nn.Sigmoid()(self.speaker_weights_a2(U_p)) * U_p

        if 'v' in self.modals:
            U_v = self.linear_v(U_v)
            emotions_v = U_v
            if self.av_using_lstm:
                emotions_v, hidden_v = self.lstm_v(U_v)
            U_, qmask_ = U_v.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_v.shape[-1]).type(U_v.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_v(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_v = torch.nn.Sigmoid()(self.speaker_weights_v1(emotions_v)) * emotions_v + torch.nn.Sigmoid()(self.speaker_weights_v2(U_p)) * U_p

        #####################################################
        ######## get features from batch data ###############
        features_l = self.simple_batch(emotions_l, lengths)
        features_a = self.simple_batch(emotions_a, lengths)
        features_v = self.simple_batch(emotions_v, lengths)
        persona = self.simple_batch(persona.transpose(0, 1), lengths)
        #####################################################


        #####################################################
        ######## get persona fusion ##########################
        qmask = torch.cat([qmask[:lengths[dia_len_index],dia_len_index,:] for dia_len_index in range(len(lengths))], dim=0)
        modal_length = features_l.shape[0] if features_l is not None else features_a.shape[0] if features_a is not None else features_v.shape[0]
        if self.wo_persona == False:
            if self.persona_transform:
                persona = self.linear_persona(persona)
            l_persona_edge = self.constructPersonaEdge(qmask=qmask, lengths=lengths) if features_l is not None else None
            a_persona_edge = self.constructPersonaEdge(qmask=qmask, lengths=lengths) if features_a is not None else None
            v_persona_edge = self.constructPersonaEdge(qmask=qmask, lengths=lengths) if features_v is not None else None
            features_l_persona = self.personaGAT_l(features_l, persona, l_persona_edge) if features_l is not None else None
            features_a_persona = self.personaGAT_a(features_a, persona, a_persona_edge) if features_a is not None else None
            features_v_persona = self.personaGAT_v(features_v, persona, v_persona_edge) if features_v is not None else None
            # features_l = features_l + torch.nn.Sigmoid()(self.persona_transformation_l(features_l_persona))*features_l_persona if features_l is not None else None
            # features_a = features_a + torch.nn.Sigmoid()(self.persona_transformation_a(features_a_persona))*features_a_persona if features_a is not None else None
            # features_v = features_v + torch.nn.Sigmoid()(self.persona_transformation_v(features_v_persona))*features_v_persona if features_v is not None else None
            features_l = torch.nn.Sigmoid()(self.persona_transformation_l1(features_l))*features_l + torch.nn.Sigmoid()(self.persona_transformation_l2(features_l_persona))*features_l_persona if features_l is not None else None
            features_a = torch.nn.Sigmoid()(self.persona_transformation_a1(features_a))*features_a + torch.nn.Sigmoid()(self.persona_transformation_a2(features_a_persona))*features_a_persona if features_a is not None else None
            features_v = torch.nn.Sigmoid()(self.persona_transformation_v1(features_v))*features_v + torch.nn.Sigmoid()(self.persona_transformation_v2(features_v_persona))*features_v_persona if features_v is not None else None


        #####################################################
        ######## cross task #################################
        features_erc, edge_index_erc, edge_indice_erc, edge_type_erc, edge_dialog_erc, modals, modal_bias = self.constructErcEdges(features_l, features_a, features_v, qmask, lengths, erc_windows=self.erc_windows)
        features_shift, edge_index_shift, edge_indice_shift, edge_type_shift, edge_dialog_shift, modals, modal_bias = self.constructShiftEdges(features_l, features_a, features_v, qmask, lengths, shift_windows=self.shift_windows, edge_index_bias=sum(lengths)*len(modals))
        features_interactive, edge_index_interactive, edge_indice_interactive, edge_type_interactive, edge_dialog_interactive = self.constructInteractiveEdges(features_erc, features_shift, qmask, lengths, modals, modal_bias, interactive_windows=self.interactive_windows)
        assert torch.equal(edge_index_erc + sum(lengths)*len(modals), edge_index_shift)
        if self.wo_crosstask == False:
            features_interactive_after = self.interactiveGAT(features_interactive, \
                                                            torch.cat([edge_index_erc, edge_index_shift, edge_index_interactive], dim=-1), \
                                                            torch.cat([edge_indice_erc, edge_indice_shift, edge_indice_interactive], dim=-1), \
                                                            torch.cat([edge_type_erc, edge_type_shift, edge_type_interactive]), \
                                                            torch.cat([edge_dialog_erc, edge_dialog_shift, edge_dialog_interactive]))
        else:
            features_interactive_after = self.interactiveGAT(features_interactive, \
                                                            torch.cat([edge_index_erc, edge_index_shift], dim=-1), \
                                                            torch.cat([edge_indice_erc, edge_indice_shift], dim=-1), \
                                                            torch.cat([edge_type_erc, edge_type_shift]), \
                                                            torch.cat([edge_dialog_erc, edge_dialog_shift]))
        # features = features_interactive + torch.nn.Sigmoid()(self.interactive_transformation(features_interactive_after))*features_interactive_after
        features =  torch.nn.Sigmoid()(self.interactive_transformation1(features_interactive)) * features_interactive + torch.nn.Sigmoid()(self.interactive_transformation2(features_interactive_after))*features_interactive_after
        features_erc = features[:sum(lengths)*len(modals)]
        features_shift = features[sum(lengths)*len(modals):]
        ###########################
        ###  process erc task  ### 
        emotions_feat = torch.cat([features_erc[modal_index*modal_length : (modal_index+1)*modal_length] for modal_index in range(len(modals))], dim=-1)
        emotions_feat = self.dropout_smax_erc(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        log_prob = self.smax_fc(emotions_feat)
        log_prob = torch.nn.functional.log_softmax(log_prob, -1)


        ###########################
        ### process shift task ### 
        emotions_feat_shift = self.constructShiftData(features_shift, qmask, lengths, modal_bias, modals)
        emotions_feat_shift = torch.cat([emotions_feat_shift[modal] for modal in modals], dim=-1)
        emotions_feat_shift = self.dropout_smax_shift(emotions_feat_shift)
        emotions_feat_shift = nn.ReLU()(emotions_feat_shift)
        log_prob_shift = self.smax_fc_shift(emotions_feat_shift)
        log_prob_shift = torch.nn.functional.log_softmax(log_prob_shift, -1)
        ###########################
        return log_prob, log_prob_shift, emotions_feat, emotions_feat_shift


    def simple_batch(self, features, lengths):
        """
        Process Mini Batch data
        """
        if features == None:
            return None
        node_features = []
        batch_size = features.shape[1]
        for j in range(batch_size):
            node_features.append(features[:lengths[j], j, :])
        node_features = torch.cat(node_features, dim=0)
        return node_features


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor

