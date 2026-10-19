import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch.nn.parameter import Parameter
from model_han_gl import OSGNN_GL

# # 求不同元路径的语义注意力
# class SemanticAttention(nn.Module):
#     def __init__(self, in_size, hidden_size=128):  # 输入为：in_size=64，hidden_size=128
#         super(SemanticAttention, self).__init__()  # in_size=隐藏层维度*注意力头数，hidden_size=128
#
#         # nn.Sequential以一种简单的方式顺序组合多个神经网络层，从而形成一个神经网络模型
#         self.project = nn.Sequential(
#             # nn.Linear创建一个线性变换层（全连接层）
#             nn.Linear(in_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 1, bias=False),
#         )
#
#     def forward(self, z):  # 输入：(N, M, D * K)，其中 N 是节点数，M 是元路径数量，D 是每个注意力头的输出特征维度，K 是注意力头的数量。
#         w = self.project(z).mean(0)  # (M, 1)  # z 通过 project 层，输出形状为 (N, M, 1)。
#         # ### mean(0)：对 N 维度取平均，得到形状为 (M, 1) 的权重 w。
#         beta = torch.softmax(w, dim=0)  # (M, 1)
#         beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1),expand的作用是将前面的变量扩展到括号中的维度
#
#         return (beta * z).sum(1)  # (N, D * K)
#
#
# # Node_Attention --> Semantic_Attention --> 两层注意力
# class HANLayer(nn.Module):
#     def __init__(
#         self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
#     ):  # 输入: 元路径子图数，输入节点特征维数，设置隐藏层维度，注意力头数，降采样
#         super(HANLayer, self).__init__()
#
#         # One GAT layer for each meta path based adjacency matrix，几条元路径就有几个gat_layer
#         self.gat_layers = nn.ModuleList()
#
#         for i in range(num_meta_paths):
#             self.gat_layers.append(
#                 # 节点注意力,基于注意力的图卷积网络
#                 GATConv(
#                     in_size,
#                     out_size,
#                     layer_num_heads,
#                     dropout,
#                     dropout,
#                     activation=F.elu,
#                 )  # 输入节点特征维数，设置隐藏层维度，注意力头数，降采样
#             )
#         self.semantic_attention = SemanticAttention(
#             in_size=out_size * layer_num_heads
#         )  # 每个元路径学习到的节点特征维度（8）*注意力头数（8）
#         self.num_meta_paths = num_meta_paths  # 元路径子图数
#
#     def forward(self, gs, h):  # 输入：元路径子图列表，特征矩阵
#         semantic_embeddings = []
#
#         for i, g in enumerate(gs):  # 据我所知，GAT用不到图的拓扑结构
#             semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
#         # ### 将所有 GAT 层的输出拼接成一个张量 semantic_embeddings，其形状为 (N, M, D * K)，
#         # ### 其中 N 是节点数，M 是元路径数量，D 是每个注意力头的输出特征维度，K 是注意力头的数量。
#         semantic_embeddings = torch.stack(
#             semantic_embeddings, dim=1
#         )  # (N, M, D * K)
#
#         return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAESF(nn.Module):
    def __init__(
        self, num_meta_paths, node_size, in_size, hidden_size, out_size, dropout, ho_adj
    ):  # 输入分别是元路径子图数，节点数量，输入节点特征维度，设置隐藏层维度，分类数，注意力头数，降采样
        super(HAESF, self).__init__()

        self.out_size = out_size
        self.in_size = in_size[0]

        self.adj = ho_adj
        self.k = self.out_size
        self.t = self.k
        self.feat_dim = self.in_size
        self.soft = nn.Softmax(dim=1)
        self.B = self.get_b_of_modularity(self.adj)  # 模块化, 邻接矩阵减去处理后的归一化度矩阵

        self.layers = nn.ModuleList()
        self.layers.append(
            OSGNN_GL(in_size, hidden_size, num_meta_paths, dropout, out_size)
        )  # 输入元路径子图数，输入节点特征维数，设置隐藏层维度，注意力头数，降采样

        """ parameters """  # 定义参数，应用Xavier均匀分布初始化
        self.topo_par = nn.Parameter(torch.FloatTensor(self.k, self.k))
        self.feat_par = nn.Parameter(torch.FloatTensor(self.t, self.feat_dim))
        self.topic = nn.Parameter(torch.FloatTensor(self.k, self.t))
        #  # 参数初始化
        torch.nn.init.xavier_uniform_(self.topo_par)
        torch.nn.init.xavier_uniform_(self.feat_par)
        torch.nn.init.xavier_uniform_(self.topic)

    def get_b_of_modularity(self, A):  # 模块化
        K = 1 / (A.sum().item()) * (
                    A.sum(axis=1).reshape(A.shape[0], 1) @ A.sum(axis=1).reshape(1, A.shape[0]))  # 头节点的度与尾节点的度相乘，再归一化
        return A - K  # 边减去度相乘，评估图中社区结构的质量

    def constraint(self):  # 对解码器中，重构拓扑结构与重构属性的中间矩阵（7,7）隐藏参数矩阵进行约束处理，使它们在一定范围内归一化。
        w = self.topo_par.data.clamp(0, 1)  # 将 self.topo_par参数的数值限制在 [0, 1] 的范围内，超出范围的数值将被截断
        col_sums = w.sum(dim=0)
        w = torch.divide(w.t(), torch.reshape(col_sums, (-1, 1))).t()  # 对W进行归一化
        self.topo_par.data = w  # 将处理后的 w 赋值回 self.topo_par.data

        w = self.topic.data.clamp(0, 1)
        col_sums = w.sum(dim=0)
        w = torch.divide(w.t(), torch.reshape(col_sums, (-1, 1))).t()
        self.topic.data = w

    #     # ## 假如节点属性维度是320，隐藏层维度是16，类别数是3
    #     self.cluster_layer1 = Parameter(torch.Tensor(node_size, hidden_size))
    #     torch.nn.init.xavier_normal_(self.cluster_layer1.data)
    #     self.cluster_layer2 = Parameter(torch.Tensor(hidden_size, out_size))
    #     torch.nn.init.xavier_normal_(self.cluster_layer2.data)
    #     self.cluster_layer3 = Parameter(torch.Tensor(out_size, self.in_size))
    #     torch.nn.init.xavier_normal_(self.cluster_layer3.data)
    #
    # # ## 语义提取
    # def get_semantic(self, feature):  # feature=Tensor(4057, 334)
    #     W1 = self.cluster_layer1  # Parameter(4057, 64)
    #     W2 = self.cluster_layer2  # Parameter(64, 4)
    #     S = self.cluster_layer3   # Parameter(4, 334)
    #
    #     # ### 公式（14）torch.div为逐元素除法
    #     W1 = torch.multiply(W1, torch.div(feature @ S.T @ W2.T, W1 @ W2 @ S @ S.T @ W2.T))
    #     W2 = torch.multiply(W2, torch.div(W1.T @ feature @ S.T, W1.T @ W1 @ W2 @ S @ S.T))
    #     S = torch.multiply(S, torch.div(W2.T @ W1.T @ feature, W2.T @ W1.T @ W1 @ W2 @ S))
    #     return S

    def forward(self, features, G, ADJ, type_mask, norm, agg, sm_adj):  # 输入：元路径子图列表，特征矩阵
        # V = self.get_semantic(features[0])  # 通过非负矩阵分解，得到社区语义矩阵

        # self.layers 有一个HANLayer,HANLayer有3个gat_layer(元路径数)
        for gnn in self.layers:
            emb, h, loss = gnn(features, G, ADJ, type_mask, norm, agg, sm_adj)

        adj_pred = self.decoder(emb)  # 结构解码，输出为重构的邻接矩阵
        emb_feat = emb @ self.topic @ self.feat_par  # reconstruction (attribute)重建(属性)，（解码器：[2708,7]*[7,7]*[7,1433]）

        return h, emb, adj_pred, self.soft(emb_feat), self.soft(emb)  # V  # 输出为节点的多注意力头维的嵌入(N, D * K)，预测标签概率，重构同构图，S

    def decoder(self, Z):  # 结构解码器
        adj_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return adj_pred
