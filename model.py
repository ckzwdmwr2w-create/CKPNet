# coding=utf-8
from fastNLP import seq_len_to_mask
import torch
from torch import nn
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
from fastNLP.modules import AvgPoolWithMask
from transformers import BertModel, BertTokenizer, BertConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics  import precision_recall_curve, precision_score, f1_score, recall_score, accuracy_score
from graph import GraphNetwork
#from contrastiveloss import ContrastEdge
from op import SubtractOp, UnionOp
from nltk.tokenize import sent_tokenize

class MainEncoder(nn.Module):
    def __init__(self, Encoder, device):
        super(MainEncoder, self).__init__()
        self.device         = device
        self.Encoder        = BertModel.from_pretrained(Encoder)
        self.tokenizer      = BertTokenizer.from_pretrained(Encoder)
        self.freeze_layers(numFreeze = 6)
        self.Encoder.to(self.device)
        self.pool               = AvgPoolWithMask()

    def freeze_layers(self, numFreeze):
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name, param in self.Encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break


    def forward(self, text, class_text = None):
        if class_text == None:
            tokenizer = self.tokenizer(
                        text,
                        padding = True,
                        truncation = True,
                        max_length = 512,
                        return_tensors='pt'  
                        )
        else:
            tokenizer = self.tokenizer(
                        text, class_text,
                        padding = True,
                        truncation = True,
                        max_length = 512,
                        return_tensors='pt'  
                        )

        input_ids = tokenizer['input_ids'].to(self.device)
        token_type_ids = tokenizer['token_type_ids'].to(self.device)
        attention_mask = tokenizer['attention_mask'].to(self.device)
        
        dict = self.Encoder(input_ids = input_ids, 
                               attention_mask = attention_mask, 
                               token_type_ids = token_type_ids, 
                               return_dict = True,
                               output_hidden_states = True)
        
        encoder_output = dict.last_hidden_state
        #hidden_states = dict.hidden_states
        #pooler_output = dict.pooler_output
        embs = self.pool(encoder_output, attention_mask)
    
        #return encoder_output, attention_mask
        return embs


'''
class BaselineEncoder(nn.Module):
    def __init__(self, Encoder, device):
        super(MainEncoder, self).__init__()
        self.device         = device
        self.Encoder        = AutoModel.from_pretrained("microsoft/infoxlm-base")
        self.tokenizer      = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
        self.freeze_layers(numFreeze = 6)
        self.Encoder.to(self.device)
        self.pool               = AvgPoolWithMask()

    def freeze_layers(self, numFreeze):
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name, param in self.Encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, text, class_text = None):
        inputs = self.tokenizer(
                    text,
                    padding = True,
                    truncation = True,
                    max_length = 512,
                    return_tensors='pt'  
                    )
        attention_mask = inputs['attention_mask'].to(self.device)
        encoder_output = self.Encoder(**inputs.to(self.device)).last_hidden_state  
        #print(outputs.size())
        #exit()
        #hidden_states = dict.hidden_states
        #pooler_output = dict.pooler_output
        embs = self.pool(encoder_output, attention_mask)
    
        #return encoder_output, attention_mask
        
        return embs

'''

'''
class MainEncoder(nn.Module):
    def __init__(self, Encoder, device):
        super().__init__()
        self.device = device

        # 初始化 SentenceTransformer
        #self.model = SentenceTransformer('sentence-transformers/LaBSE')
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.Encoder = self.model[0].auto_model  # 内部 transformer
        self.tokenizer = self.model.tokenizer    # tokenizer 从 SentenceTransformer 拿

        # 冻结前几层
        #self.freeze_layers(numFreeze = 6)

        # 移动模型到指定 device
        self.Encoder.to(self.device)
        self.pool               = AvgPoolWithMask()
        print('****************************MiniLM*****************************')

    def freeze_layers(self, numFreeze):
        # 确保 numFreeze 是整数

        # pooler 和后几层不冻结
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer." + str(i))

        for name, param in self.Encoder.named_parameters():
            param.requires_grad = False  # 默认冻结
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, text):
        """
        text: list[str], 支持多语言 batch
        返回 CLS 向量，可微调
        """
        # tokenizer 编码
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        attention_mask = inputs['attention_mask'].to(self.device)
        encoder_output = self.Encoder(**inputs.to(self.device)).last_hidden_state  
        #print(outputs.size())
        #exit()
        #hidden_states = dict.hidden_states
        #pooler_output = dict.pooler_output
        embs = self.pool(encoder_output, attention_mask)
    
        #return encoder_output, attention_mask
        
        return embs

'''

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.W1 = torch.nn.Linear(768, 768, bias= True)
        self.W2 = torch.nn.Linear(768, 1, bias= True)
        self.tanh = torch.nn.Tanh()

    def forward(self, imput_tensor):
        A  = self.W2(self.tanh(self.W1(imput_tensor)))
        output_tensor = torch.matmul(A.T, imput_tensor)  # torch.Size([1, 768])
            
        return output_tensor



class SelfAttentive(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttentive, self).__init__()
        
        self.linear1 = nn.Linear(hidden_dim, int(hidden_dim/3))
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(4*hidden_dim, hidden_dim)
        self.relu = nn.LeakyReLU(0.3, inplace=True)
        self.context_vector = nn.Parameter(torch.randn(int(hidden_dim/3), 4), requires_grad=True)
    
    def forward(self, last_hidden_states):
        b, _, _ = last_hidden_states.shape
        vectors = self.context_vector.unsqueeze(0).repeat(b, 1, 1)
        
        h = self.linear1(last_hidden_states) # (b, t, h)
        h = self.tanh(h)
        scores = torch.bmm(h, vectors) # (b, t, 4)
        scores = nn.Softmax(dim=1)(scores) # (b, t, 4)
        outputs = torch.bmm(scores.permute(0, 2, 1), last_hidden_states).view(b, -1) # (b, 4h)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        return outputs

   
class MultiDimensionAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_dimensions = 1):
        super(MultiDimensionAttention, self).__init__()
        self.num_dimensions = num_dimensions
        
        # Attention机制
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=True)  # 特征变换
        self.W2 = nn.Linear(hidden_dim, num_dimensions, bias=True)  # 生成多个维度的attention
        self.tanh = nn.Tanh()
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=True) 
        
    def forward(self, sent_embs):
        # sent_embs: [26, 768]
        
        # 通过attention机制生成多个维度的权重
        hidden = self.tanh(self.W1(sent_embs))  # [26, 768]
        attention_logits = self.W2(hidden)      # [26, 5] - 每个句子对每个维度的重要性
        
        # 对每个维度分别进行softmax
        #P = F.softmax(attention_logits, dim=0)  # [26, 5] - 每列是一个维度的attention权重
        
        # 生成每个维度的表征
        proto_node_feat = torch.matmul(attention_logits.transpose(0, 1), sent_embs)  # [5, 768]
        
        return proto_node_feat  # [5, 768] - 5个维度的表征


class main_model(nn.Module):
    def __init__(self, args):
        super(main_model, self).__init__()
        print('*********************Ours************************')
        self.device             = args.device
        self.Encoder            = MainEncoder(args.Encoder, args.device)
        #self.hidden_dim         = 768
        self.hidden_dim         = 768
        self.multi_sem_att      = MultiDimensionAttention(input_dim = self.hidden_dim, hidden_dim = self.hidden_dim)
        self.gnn_module         = GraphNetwork(args)
        self.subtract_op        = SubtractOp(input_dim = 768, S_layers_num = 2).to(self.device)
        self.union_op           = UnionOp(input_dim = 768, U_layers_num = 2).to(self.device)
        self.criterion          = nn.MSELoss(reduction = "mean")
        self.loss_ce            = nn.CrossEntropyLoss()
        self.attention          = SelfAttention()
        self.pool               = AvgPoolWithMask()
        self.classifier_ce         = nn.Sequential(
                nn.Linear(768, 6, bias = True),#情感只有四个
        )
        self.classifier_mse        = nn.Sequential(
                nn.Linear(768, 1, bias = True),#情感只有四个
        )
        self.mapping_layer = nn.Linear(self.hidden_dim, 6)
        self.mu = 1e-4
        self.e  = -0.05
        self.n_classes = 6
        self.n_samples = args.batch_size
        self.n_distances = self.n_classes - 1
        self.distances = torch.zeros(self.n_distances, device = self.device, dtype = torch.float64)
        
        for i in range(self.n_distances):
            self.distances[i] = self.__inverse_softplus(0.5 + torch.rand(1) * 0.5)
        self.distances = nn.Parameter(self.distances[:])
        
    def __inverse_softplus(self, t):
        # to get the inverse of softplus when setting margins
        return torch.where(t > 20, t, torch.log(torch.exp(t) - 1))
    def label2edge(self, label):
        num_samples = label.size(0)
        label_i = label.unsqueeze(-1).repeat(1, num_samples)
        label_j = label_i.transpose(0, 1)
        # compute edge
        true_edge = torch.eq(label_i, label_j).to(torch.double).to(self.device)
        #print(label)
        #print(true_edge)
        return true_edge
    def get_edge(self, init_edge, src_num, tgt_num):
        init_edge[src_num:, :] = 0
        init_edge[:, src_num:] = 0
        '''
            init_edge:  tensor([[[1,   1,   1,   0,   0,   0,   0.5, 0.5],
                                 [1,   1,   1,   0,   0,   0,   0.5, 0.5],
                                 [1,   1,   1,   0,   0,   0,   0.5, 0.5],
                                 [0,   0,   0,   1,   1,   1,   0.5, 0.5],
                                 [0,   0,   0,   1,   1,   1,   0.5, 0.5],
                                 [0,   0,   0,   1,   1,   1,   0.5, 0.5],
                                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],

                                [[0,  0,   0,   1,   1,   1,   0.5, 0.5],
                                 [0,   0,   0,   1,   1,   1,   0.5, 0.5],
                                 [0,   0,   0,   1,   1,   1,   0.5, 0.5],
                                 [1,   1,   1,   0,   0,   0,   0.5, 0.5],
                                 [1,   1,   1,   0,   0,   0,   0.5, 0.5],
                                 [1,   1,   1,   0,   0,   0,   0.5, 0.5],
                                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]])
        '''
        return init_edge
    '''
    def forward(self, batch):
        node_feat = self.Encoder(batch['essay']) 
   
        #print('node_feat: ', node_feat.size())
        true_edge = self.label2edge(batch['score'])
        node_feat_updated, edge_feat, inter_feat = self.gnn_module(node_feat)    # [5, 768]        
        # 重构
        specific_feat = self.subtract_op(node_feat, node_feat_updated)
        node_feat_rec_v1, node_feat_rec_v2 = self.union_op(specific_feat, node_feat_updated)
        loss_rec = (self.criterion(node_feat, node_feat_rec_v1) + self.criterion(node_feat, node_feat_rec_v2)) / 2
        
        # 正交
        inter_norm = F.normalize(node_feat_updated, dim = -1)
        specific_norm = F.normalize(specific_feat, dim = -1)
        dot = torch.matmul(inter_norm, specific_norm.T)
        loss_orth = torch.mean(dot ** 2)
        #print('node_feat_updated: ', node_feat_updated.size())
        #print('edge_feat: ', edge_feat.size())
        #print('inter_feat: ', inter_feat.size())
        #print('specific_feat: ', specific_feat.size())
        #print('node_feat_rec_v1: ', node_feat_rec_v1.size())
        #print('node_feat_rec_v2: ', node_feat_rec_v2.size())
        preds = self.classifier(node_feat_updated).squeeze(-1)
        #loss_aux = F.mse_loss(self.classifier(node_feat), torch.tensor(batch['score'].float()).to(self.device))
        loss_aux = self.criterion(edge_feat, true_edge)
        #preds = self.classifier(node_feat).squeeze(-1)
        #loss_orth, loss_rec, loss_aux = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)
        #print(preds.size())
        #exit()
        return preds, loss_orth, loss_rec, loss_aux

    
    def forward(self, batch):
        batch_embs = torch.tensor([]).to(self.device)
        for essay in batch['essay']:
            sent_splits = sent_tokenize(essay)
            sent_hidden, sent_mask = self.Encoder(sent_splits) 
            sent_embs = self.pool(sent_hidden, sent_mask) # [26, 768]
            #print(sent_embs.size())
            #node_feat_updated, _, _ = self.gnn_module(sent_embs)
            #embs = node_feat_updated.mean(dim = 0).unsqueeze(0)
            #print(embs.size())
            embs = self.multi_sem_att(sent_embs) # [6, 768]
            batch_embs = torch.cat((batch_embs, embs), dim = 0)
        #print(batch_embs.size())
        #encoder_output, attention_mask = self.Encoder(batch['essay']) 
        #essay_feat = self.pool(encoder_output, attention_mask)
        #node_feat_updated, edge_feat, inter_feat = self.gnn_module(essay_feat)    # [5, 768] 
        #feat = self.NET(torch.cat([batch_embs, node_feat_updated], dim = -1))
        preds = self.classifier(batch_embs).squeeze(-1)
        loss_orth, loss_rec, loss_aux = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)

        #exit()
            #sem_dim_embs = F.relu(sem_dim_embs)
            #sem_dim_embs = F.relu(sem_dim_embs)
        #embs = dim_embs.view(1, -1)
        return preds, loss_orth, loss_rec, loss_aux

    '''
    
    
    def compute_ldc_entropy(self, h):
        logits = self.mapping_layer(h)
        probs = F.softmax(logits, dim=1)
        #
        num_classes = probs.size(-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim = -1)
        max_entropy = torch.log(torch.tensor(num_classes, dtype = torch.float))
        # 归一化熵: 0(完全确定) 到 1(完全均匀)
        ldc = 1 - entropy / max_entropy
        
        return ldc
    
    
    def compute_ldc(self, node_features):
        logits = self.mapping_layer(node_features)
        class_probs = F.softmax(logits, dim=1)
        max_prob = torch.max(class_probs, dim=1)[0]  # [batch_size]
        min_prob = torch.min(class_probs, dim=1)[0]  # [batch_size]
        ldc = max_prob - min_prob
        return ldc
    
    def inter_layer_filtering_gate(self, ldc_prev, ldc_current):
        phi = (ldc_current > ldc_prev + self.e).float()
        return phi
    
    def apply_filtering(self, h_prev, h_current, phi):
        phi = phi.unsqueeze(1)  # [batch_size, 1] 用于广播
        h_filtered = (1 - phi) * h_prev + phi * h_current
        return h_filtered
    
    def initial_compensation(self, h_initial, h_current, ldc_initial, ldc_current):
        # 计算补偿权重Λ
        lambda_weights = ldc_current / (ldc_initial + ldc_current + self.mu)
        lambda_weights = lambda_weights.unsqueeze(1)  # [batch_size, 1]
        # 应用补偿
        h_compensated = (1 - lambda_weights) * h_initial + lambda_weights * h_current
        
        return h_compensated
    
    def compensation(self, init_feat, init_ldc, cur_feat, cur_ldc, phi):
        phi = phi.unsqueeze(1)
        final_feat = torch.where(
            phi.bool(),
            # 采用当前层时：进行初始补偿
            self.initial_compensation(h_initial = init_feat, h_current = cur_feat, ldc_initial = init_ldc, ldc_current = cur_ldc),
            # 保留前一层时：不需要补偿
            init_feat
        )
        return final_feat
    
    def Graph(self, init_feat, init_ldc, down_feat):
        down_ldc = self.compute_ldc_entropy(down_feat)
        up_feat, edge_feat = self.gnn_module(down_feat)
        up_ldc = self.compute_ldc_entropy(up_feat)
        #print('*' * 20)
        #print(down_ldc)
        #print(up_ldc)
        #print('*' * 20)
        phi = self.inter_layer_filtering_gate(ldc_prev = down_ldc, ldc_current = up_ldc)
        filter_feat = self.apply_filtering(h_prev = down_feat, h_current = up_feat, phi = phi)
        final_feat = self.compensation(init_feat, init_ldc, cur_feat = filter_feat, cur_ldc = up_ldc, phi = phi)
        return final_feat, edge_feat, down_ldc, up_ldc
        #return filter_feat, edge_feat, down_ldc, up_ldc
    
    def Contrastive(self, cos_sim, target):
        '''
        label_diff 也是 (N,N)：为 0 表示同类。
        positives 为 bool 矩阵表示「同类对」；为了不把样本自身视作正对，把对角设为 False。
        negatives = ~positives 表示不同类对（注意这里对角会变为 True —— 这是后面要注意的潜在问题）。
        '''
        label_diff = torch.abs(target.unsqueeze(0) - target.unsqueeze(1)).float()
        positives = label_diff <= 0
        negatives = ~positives
        positives[torch.eye(self.n_samples).bool()] = False # to avoid taking data point itself as a positive pair
        '''
        pos_cossim[i,j] 在是正对的位置保留真实余弦，其它位置为 +inf
        neg_cossim[i,j] 在是负对的位置保留真实余弦，其它位置为 -inf
        '''
        pos_cossim, neg_cossim = cos_sim.clone(), cos_sim.clone()
        pos_cossim[~positives] = torch.inf # setting false of positive tensor to inf
        neg_cossim[~negatives] = -torch.inf # setting false of negative tensor to -inf
        
        # deriving the distance matrix
        '''
        pos_distances 保证类别之间的间隔是正数，
        ### self.distances = [0.2, 1.5]， pos_distances = [0.798, 1.702]
        class_positions 通过累计和把每个类放到一条数轴上。
        ### class_positions = [0.0, 0.798, 2.500]
        distance_matrix 是类中心之间的绝对距离矩阵（对称，主对角为 0）。
        ### distance_matrix =
            [[0.000, 0.798, 2.500],
            [0.798, 0.000, 1.702],
            [2.500, 1.702, 0.000]]
        '''
        pos_distances = F.softplus(self.distances)
        class_positions = torch.cumsum(torch.cat([torch.tensor([0.0]).to(self.device), pos_distances]), dim = 0)
        distance_matrix = torch.abs(class_positions.unsqueeze(0) - class_positions.unsqueeze(1))
        # assigning the margins
        '''
        假设有3个类，有4个样本：target = [0, 0, 1, 2]
        这里构造 label_indices 使得 margins 成为一个 (N,N) 矩阵：
        每一行都等于一个target
           [[0, 0, 1, 2],
            [0, 0, 1, 2],
            [0, 0, 1, 2],
            [0, 0, 1, 2]]
        margins[i,j] = distance_between_classes(label_j, label_i)。
        然后把非负对（即正对）的位置 margin 设为 0，因为 margin 只对“负样本比较”有意义。
        margins =
                [[0.000, 0.000, 0.798, 2.500],
                [0.000, 0.000, 0.798, 2.500],
                [0.798, 0.798, 0.000, 1.702],
                [2.500, 2.500, 1.702, 0.000]]
        '''
        label_indices = target.unsqueeze(0).repeat(self.n_samples, 1).to(self.device)
        margins = distance_matrix[label_indices.long(), label_indices.long().t()]
        margins[~negatives] = 0

        mean_n_pair_loss = torch.tensor([]).to(self.device) # to collect n-pair loss for each column in positive tensor
        loss_masks_2 = torch.tensor([]).to(self.device)
        '''
        cos_sim =
            [[1.00, 0.90, 0.20, 0.10],
            [0.90, 1.00, 0.15, 0.05],
            [0.20, 0.15, 1.00, 0.60],
            [0.10, 0.05, 0.60, 1.00]]
        positives =
            [[F, T, F, F],
            [T, F, F, F],
            [F, F, F, F],
            [F, F, F, F]]
        pos_cossim =
            [[+inf, 0.90, +inf, +inf],
            [0.90, +inf, +inf, +inf],
            [+inf, +inf, +inf, +inf],
            [+inf, +inf, +inf, +inf]]
        negatives =
            [[T, F, T, T],
            [F, T, T, T],
            [T, T, T, T],
            [T, T, T, T]]
        neg_cossim =
            [[1.00, -inf, 0.20, 0.10],
            [-inf, 1.00, 0.15, 0.05],
            [0.20, 0.15, 1.00, 0.60],
            [0.10, 0.05, 0.60, 1.00]]
        '''
        for pos_col in pos_cossim.T: # comparing each column of positive tensor with all columns of negative tensor
            n_pair_loss = (-pos_col + neg_cossim.T + margins.T).T

            # creating masks on elements n-pair loss was calculated for accurate mean calculation, otherwise final loss will be too low 
            loss_mask1 = ~torch.isinf(n_pair_loss) # -inf indicates it's not a pos/neg pair, avoiding those;
            loss_mask2 = loss_mask1.sum(dim=1)
            n_pair_loss = F.relu(n_pair_loss)

            # Compute the row-wise mean of the masked elements
            n_pair_loss = (n_pair_loss * loss_mask1).sum(dim=1) / loss_mask2.clamp(min=1)  # Use clamp to avoid division by zero

            loss_masks_2 = torch.cat((loss_masks_2, loss_mask2.reshape(1,-1)), dim=0)
            mean_n_pair_loss = torch.cat((mean_n_pair_loss, n_pair_loss.reshape(1,-1)), dim=0)

        mean_n_pair_loss = (mean_n_pair_loss * loss_masks_2.bool()).sum(dim=0) / loss_masks_2.sum(dim=0).clamp(min=1)
        return mean_n_pair_loss.mean()
        
    def forward(self, batch):
        init_feat = self.Encoder(batch['essay']) # torch.Size([16, 768])
        if self.training:
            self.e  = -0.1
        else:
            self.e  = 0
        init_ldc = self.compute_ldc_entropy(init_feat)
        #print('init_ldc: ', init_ldc.size())
        #exit()
        cur_feat, edge_feat, down_ldc, up_ldc = self.Graph(init_feat = init_feat, init_ldc = init_ldc, down_feat = init_feat)
        all_ldcs = []
        all_ldcs.append(init_ldc)  # 初始层
        all_ldcs.append(down_ldc)         # 第1层 
        all_ldcs.append(up_ldc)         # 第2层
        all_ldcs_tensor = torch.stack(all_ldcs, dim=0)  # [4, 16]
        gdc = torch.mean(all_ldcs_tensor)  # 标量
        
        #exit()
        
        label_ce = torch.tensor(batch['score'].long().to(self.device) - 1)
        logits = self.classifier_ce(init_feat)
        preds = torch.argmax(logits, -1) + 1
        loss_ce = self.loss_ce(logits, label_ce)
        
        #print(preds)
        #exit()
        label_ms = torch.tensor(batch['score'].float().to(self.device))
        preds = self.classifier_mse(cur_feat).squeeze(-1)
        loss_ms = F.mse_loss(preds.float(), label_ms)
        # 
        loss_cl = self.Contrastive(cos_sim = edge_feat, target = label_ms - 1) # target从0 开始
        #print(loss_cl)
        #exit()
        #loss_aux = torch.tensor(0.0).to(self.device)
        return preds, loss_ms, gdc, loss_cl, loss_ce

            
        

        
