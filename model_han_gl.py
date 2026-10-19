import torch
import torch.nn as nn
import torch.nn.functional as F
from model_oslayer import OSGNN
import numpy as np

class OSGNN_GL(nn.Module):
    def __init__(self, input_dim, feat_hid_dim, metapath, dropout, outsize):
        super(OSGNN_GL, self).__init__()
        # ## input_dim: 每种类型节点的属性维度; feat_hid_dim: 隐藏层维度; meta-path: 元路径数; outsize: 分类标签的类别数.
        self.dropout = nn.Dropout(p=dropout)
        self.metapath = metapath
        self.feat_hid_dim = feat_hid_dim
        self.non_linear = nn.ReLU()
        self.feat_mapping = nn.ModuleList([nn.Linear(m, feat_hid_dim, bias=True) for m in input_dim])  # 使用线性全连接进行特征投影
        for fc in self.feat_mapping:
            nn.init.xavier_normal_(fc.weight, gain=1.414)  # 对每个线性层 fc 的权重进行 Xavier 正态分布的初始化
            # nn.init.xavier_normal_(fc.weight, gain=0.8)  # 对每个线性层 fc 的权重进行 Xavier 正态分布的初始化
        self.overall_graph_gen = GraphChannelAttLayer(metapath)  # 创建图注意力的图神经网络层
        # ## 创建本文模型消息传递，包含多图操作的神经网络模型
        self.het_graph_encoder_anchor = OSGNN(num_graph=3, hidden_size=feat_hid_dim, out_size=outsize, num_layer=1, dropout=dropout)


    def forward(self, features, G, ADJ, type_mask, norm, agg, sm_adj):

        transformed_features = torch.zeros(type_mask.shape[0], self.feat_hid_dim).to(features[0].device)  # 初始化一个零张量
        for i, fc in enumerate(self.feat_mapping):  # 分别对不同类型的节点特征进行映射，fc是（线性变换+ReLU非线性）的模块
            node_indices = np.where(type_mask == i)[0]  # 提取出各种类型的节点索引
            transformed_features[node_indices] = fc(features[i])  # 存储映射后的节点特征
        h = transformed_features
        feat_map = self.dropout(h)  # 应用 Dropout，随机失活部分特征。

        # Subgraph generation of target node。生成目标节点的子图
        new_G = self.overall_graph_gen(G, norm[1])
        new_G = new_G.t() + new_G
        new_G = F.normalize(new_G, dim=1, p=norm[0])  # 对 new_G 进行归一化处理

        # G = [ADJ, new_G]
        G = [ADJ, new_G, sm_adj]  # 构造包含原始邻接矩阵和新生成图的列表 G

        logits, h, loss = self.het_graph_encoder_anchor(G, feat_map, agg)  # 编码logits: Tensor(4019, 3); h: Tensor(4019, 64).
        return logits, h, loss


class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))  # 创建了一个可学习的参数 self.weight
        nn.init.constant_(self.weight, 0.1)  # 初始化 self.weight 的值为常数 0.1
    def forward(self, adj_list, norm_2):
        adj_list = torch.stack(adj_list)  # 将传入的 adj_list 列表中的每个邻接矩阵堆叠成一个张量 adj_list
        if norm_2 != 0:
            adj_list = F.normalize(adj_list, dim=1, p=norm_2)
            # 使用 PyTorch 的 F.normalize 函数对 adj_list 进行标准化。这里的 dim=1 表示对第一个维度（通道维度）进行标准化，p=norm_2 表示使用 norm_2 范数进行标准化
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)  # 计算并返回加权邻接矩阵的加权和


