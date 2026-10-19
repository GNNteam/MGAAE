import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch.nn.parameter import Parameter
import math

device = torch.device("cuda:0")
# ## 这段介绍的是Hilbert-Schmidt Independence Criterion (HSIC)希尔伯特-施密特独立性准则，类似于互信息，用来衡量两个变量之间的独立性
# 计算两组特征向量 emb1 和 emb2 之间的 HSIC 值，用以度量它们之间的依赖关系
def loss_dependence(emb1, emb2, dim):
    # R为中心化核矩阵（即消除均值效应）
    #  # R 表示中心化矩阵，通过单位矩阵减去均匀分布矩阵，实现了对数据均值的去中心化操作。这对 HSIC 的计算非常重要，因为 HSIC 是对数据均值不敏感的度量。
    R = torch.eye(dim).to(device) - (1/dim) * torch.ones(dim, dim).to(device)  # [3025, 3025]
    K1 = torch.mm(emb1, emb1.t())  # [3025, 3025]emb1.shape=[3025, 256]
    # ## 计算 emb1 的核矩阵 K1，即 emb1 与其转置 emb1.t() 的矩阵乘积。K1 是一个 dim x dim 的矩阵，表示特征向量之间的相似性
    K2 = torch.mm(emb2, emb2.t())  # 外积，形成了核矩阵。这些核矩阵衡量了样本之间的相似性，是 HSIC 计算的基础。
    RK1 = torch.mm(R, K1)  # [3025, 3025]
    # ## 将中心化矩阵 R 应用于核矩阵 K1，得到 RK1。这是通过矩阵乘法实现的，RK1 是中心化后的 K1
    RK2 = torch.mm(R, K2)  # 经过中心化处理后的核矩阵。
    HSIC = torch.trace(torch.mm(RK1, RK2))  # 公式14, 计算RK1和RK2的矩阵乘积. 计算矩阵的迹，即矩阵对角线上元素的和。
    # ## 计算 RK1 和 RK2 的矩阵乘积，然后取其迹（即对角线元素的和）。这是 HSIC 的计算方式，它度量了两个中心化核矩阵之间的相似性。
    return HSIC  # 是一种用于测量两个随机变量之间独立性的统计量。HSIC 值越高，表示两个变量之间的依赖关系越强。这个函数计算的是在给定特征向量集上的 HSIC 值，用于研究它们之间的依赖关系。

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()  # 定义了一个序列模块，32 -> 128  -> 1
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z, agg):  # z: Tensor(4019, 2, 32)
        # 拼接
        if agg == 'contact':
            beta = torch.Tensor([[1], [0], [0]]).cuda()  # 输出Tensor(2,1)
            beta = beta.expand((z.shape[0],) + beta.shape)  # 将其扩展为 (z.shape[0], 2, 1) 的形状
            z1 = (beta * z).sum(1)  # 取出原始图的嵌入：Tensor(4019,32)，32维
            alpha = torch.Tensor([[0], [1], [0]]).cuda()
            alpha = alpha.expand((z.shape[0],) + alpha.shape)
            z2 = (alpha * z).sum(1)  # 取出元路径图的嵌入：32维
            rama = torch.Tensor([[0], [0], [1]]).cuda()
            rama = rama.expand((z.shape[0],) + rama.shape)
            z3 = (rama * z).sum(1)  # 取出元路径图的嵌入：32维
            z = torch.cat((z1, z2, z3), 1)  # 沿着维度 1 进行拼接，得到最终的 z：64维
        # 注意力
        elif agg == 'att':
            w = self.project(z).mean(0)  # 通过self.project(z)计算每个节点的注意力权重，得到（4019,3,1）；然后通过.mean(1)对1维进行平均
            beta = torch.softmax(w, dim=0)  # Tensor(3, 1)
            # print(beta)
            # beta=torch.tensor([[0.333], [0.333], [0.333]])  # 示例
            beta = beta.expand((z.shape[0],) + beta.shape)  # Tensor(4019, 3, 1)
            z = (beta * z).sum(1)
        # 平均
        elif agg == 'mean':
            beta = torch.tensor([[1 / 3], [1 / 3], [1 / 3]]).cuda()  # 示例均等权重
            beta = beta.expand((z.shape[0],) + beta.shape)  # 将 beta 扩展为 (z.shape[0], 3, 1) 的形状
            z = (beta * z).sum(1)
        return z


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  # 输入特征的维度
        self.out_features = out_features    # 输出特征的维度
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # 权重矩阵，形状为 (in_features, out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # 如果使用偏置项，初始化偏置向量，形状为 (out_features)
        else:
            self.register_parameter('bias', None)   # 如果不使用偏置项，则注册一个空的 bias 参数
        self.reset_parameters()   # 调用 reset_parameters 方法初始化权重和偏置

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))   # 计算标准差，用于权重初始化
        self.weight.data.uniform_(-stdv, stdv)    # 对权重矩阵进行均匀分布的初始化
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  # 如果有偏置项，则对偏置向量进行均匀分布的初始化

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)   # 执行稀疏矩阵乘法，计算输入特征与权重的乘积64 ->32
        output = torch.spmm(adj, support)       # 执行稀疏矩阵乘法，计算邻接矩阵 adj 与支持特征的乘积
        if self.bias is not None:
            return F.elu(output + self.bias)     # 如果有偏置项，对输出加上偏置并使用 ELU 激活函数
        else:
            return F.elu(output)       # 否则，仅使用 ELU 激活函数



class OSGNNLayer(nn.Module):

    def __init__(self, num_graph, in_size, out_size, dropout):
        super(OSGNNLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()  # 创建一个空的 ModuleList
        self.dropout = nn.Dropout(p=dropout)
        for i in range(num_graph):  # 循环添加图卷积层
            if i == 0:  # 对于第一个图卷积层，使用 GATConv 创建。GATConv 是一个图注意力机制的卷积层，可以处理具有不同权重的图结构。
                self.gcn_layers.append(GATConv(in_size, out_size, 1, dropout, dropout, activation=F.elu, allow_zero_in_degree=True))
            else:
                self.gcn_layers.append(GraphConvolution(in_size, out_size))  # 图卷积自定义GCN，用于生成元路径图的嵌入
        self.semantic_attention = SemanticAttention(in_size=out_size)  # 自定义语义注意力层
        self.num_graph = num_graph

    def forward(self, gs, h, agg):  # gs:[DGLGraph(原始邻接矩阵), Tensor(一个聚合后的元路径图)]
        semantic_embeddings = []
        for i, g in enumerate(gs):  # 循环遍历图列表 gs
            if i == 0:
                semantic_embeddings.append(self.gcn_layers[0](gs[0], h).flatten(1)[:len(gs[1]), :])  # GATConv图卷积层对原始图计算
            else:  # 其实就是使用GCN对元路径图卷积
                semantic_embeddings.append(self.gcn_layers[i](h[:len(g), :], g).flatten(1))  # 自定义卷积层对聚合后的元路径图计算
        loss = loss_dependence(semantic_embeddings[0], semantic_embeddings[1], dim=semantic_embeddings[0].shape[0])  # 计算两个变量之间的依赖损失
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        # 语义嵌入堆叠.[Tensor(4019, 32),Tensor(4019, 32)] -> Tensor(4019, 2, 32)
        return self.semantic_attention(semantic_embeddings, agg), loss  # 同时返回计算得到的输出和损失

class OSGNN(nn.Module):
    def __init__(self, num_graph, hidden_size, out_size, num_layer, dropout):
        super(OSGNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(OSGNNLayer(num_graph, hidden_size, 32, dropout))  # 自定义注意力卷积神经网络层，输出为32维
        for l in range(1, num_layer):  # 叠加自定义的卷积层
            self.layers.append(OSGNNLayer(num_graph, hidden_size, hidden_size, dropout))
        self.predict = nn.Linear(32, out_size)  # 创建一个线性层 nn.Linear，将输入大小为 32*2，输出大小为 out_size

    def forward(self, g, h, agg):  # g:[DGLGraph(原始邻接矩阵), Tensor(一个聚合后的元路径图)], h: 节点特征：11246*64
        # a = 0
        # h1 = h[:len(g[1]), :]  # 变量初始化, 取前4019个节点的所有特征，即目标节点(节点类型映射后的节点属性)
        loss = 0  # 损失初始化
        for gnn in self.layers:  # 循环遍历层
            h, l = gnn(g, h, agg)  # 层的前向传播，对当前层 gnn 进行前向传播，更新 h
            loss = loss + l  # 计算当前层的损失 l，并将其加到 loss 中
        # 残差链接，实际上a=0, 并没有用残差链接，仅仅最终嵌入
        # h = (1 - a) * h + a * h1
        return self.predict(h), h, loss

        # # 返回经过线性层 self.predict 处理后的 h，以及 h 本身和累计的损失 loss