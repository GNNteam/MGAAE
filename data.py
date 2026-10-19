import numpy as np
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
import random
import datetime
import os

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# set the result storage path
def setup_log_dir(args):
    dt = datetime.datetime.now()
    date_postfix = "parameters/{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )
    log_dir = os.path.join(
        args["log_dir"], "{}_{}".format(args["dataset"], date_postfix)
    )
    os.makedirs(log_dir)
    print("Created directory {}".format(log_dir))
    return log_dir

def setup(args):
    set_random_seed(args["seed"])
    args["dataset"] = "ACM3025"
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    args["log_dir"] = setup_log_dir(args)
    if args["dataset"] == "DBLP":
        args["num_epochs"] = 800
        args["lr"] = 0.005
        args["patience"] = 100
        args["α"] = 50
        args["b"] = 150
        args["c"] = 0.00008
    elif args["dataset"] == "ACM3025":
        args["num_epochs"] = 800
        args["lr"] = 0.01
        args["patience"] = 10
        args["α"] = 10
        args["b"] = 5
        args["c"] = 0.00001
    elif args["dataset"] == "ACM4025":
        args["patience"] = 10
        args["α"] = 5
        args["b"] = 100
        args["c"] = 0.00008
    elif args["dataset"] == "ACM4019":
        args["patience"] = 10
        args["α"] = 5
        args["b"] = 100
        args["c"] = 0.00008
    elif args["dataset"] == "yelp":
        args["num_epochs"] = 3
        args["lr"] = 0.01
        args["patience"] = 100
        args["α"] = 0
        args["b"] = 0
        args["c"] = 1

    return args


# load data.
def load_ourdata(dataset):
    if dataset == "DBLP":
        return load_DBLP_data()
    elif dataset == "ACM3025":
        return load_ourACM3025()
    elif dataset == "ACM4025":
        return load_ACM4025()
    elif dataset == "ACM4019":
        return load_ACM_data()
    elif dataset == "yelp":
        return load_YELP()
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))

def load_ACM_data(prefix=r'.\Dataset\ACM'):
    # ####加载了三个稀疏矩阵文件 (features_0.npz, features_1.npz, features_2.npz)，并将它们转换为稠密数组格式 (toarray() 方法)
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()  # 4019*4000
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()  # 7167*4000
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()  # 60*4000

    labels = np.load(prefix + '/labels.npy')  # 加载标签数据，4019个：{0, 1, 2}

    # ## 构建特征相似图
    # sm_adj = knn_graph(features_0, 300)  # ## 根据相似度节点的数量生成相似图
    sm_adj = sim_graph(features_0, 0.3)  # ## 根据相似度阈值生成相似图

    nd_PAP = scipy.sparse.load_npz(prefix + '/pap.npz').A  # 加载稀疏矩阵，.A将稀疏矩阵转为稠密矩阵
    nd_PSP = scipy.sparse.load_npz(prefix + '/psp.npz').A

    # nd_PAP[nd_PAP != 0] = 1
    # nd_PSP[nd_PSP != 0] = 1

    PAP = nd_PAP
    PSP = nd_PSP

    ndarray_list = [PAP, PSP]
    gs = [torch.tensor(arr, dtype=torch.float32) for arr in ndarray_list]

    # adj = 0.55 * PAP + 0.45 * PSP
    adj = PAP + PSP
    adj = torch.Tensor(adj)
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))  # 5、同构图

    #  #加载并处理图的邻接矩阵
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')  # 加载稀疏邻接矩阵数据，节点数：11246；边数：
    ADJ = dgl.DGLGraph((ADJ + (ADJ.T)) / 2)  # 将其转换为 dgl.DGLGraph 对象
    # ADJ = dgl.graph(ADJ + (ADJ.T))
    ADJ = dgl.remove_self_loop(ADJ)  # remove_self_loop 和 add_self_loop 方法分别移除和添加自环边，以确保每个节点都有一个自环。
    ADJ = dgl.add_self_loop(ADJ)

    features_0 = torch.FloatTensor(features_0)  # 将ndarray特征数据格式转换为Tensor格式
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = [features_0, features_1, features_2]

    labels = torch.LongTensor(labels)
    type_mask = np.load(prefix + '/node_types.npy')  # 加载节点类型，11246个：{0, 1, 2}

    num_classes = 3

    return gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj

def load_IMDB_data(prefix=r'.\Dataset\IMDB'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    MAM = np.load(prefix + '/mam.npy')
    MAM = torch.from_numpy(MAM).type(torch.FloatTensor)
    MAM = F.normalize(MAM, dim=1, p=2)

    MDM = np.load(prefix + '/mdm.npy')
    MDM = torch.from_numpy(MDM).type(torch.FloatTensor)
    MDM = F.normalize(MDM, dim=1, p=2)

    G = [MAM, MDM]

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')
    ADJ = dgl.DGLGraph(ADJ + (ADJ.T))
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = [features_0, features_1, features_2]

    labels = torch.LongTensor(labels)
    type_mask = np.load(prefix + '/node_types.npy')

    num_classes = 3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return G, ADJ, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask

from sklearn.metrics.pairwise import cosine_similarity


def degree_normalize(adj_matrix):
    # 计算度
    degree = adj_matrix.sum(dim=1)

    # 防止度为0的节点（避免除以0）
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # 替换无穷大的值为0

    # 创建对角矩阵
    D_inv_sqrt = torch.diag(degree_inv_sqrt)

    # 对称度归一化
    normalized_adj = torch.matmul(D_inv_sqrt, torch.matmul(adj_matrix, D_inv_sqrt))

    return normalized_adj

# ## 根据相似度节点的数量生成相似图
def knn_graph(feat, topk, weight=False, loop=True):  # 来源于 SDCN
    # 计算相似度矩阵
    sim_feat = cosine_similarity(feat)
    sim_matrix = np.zeros(shape=(feat.shape[0], feat.shape[0]))

    # 确定每个节点的最近邻节点索引
    for i in range(sim_feat.shape[0]):
        ind = np.argpartition(sim_feat[i, :], -(topk + 1))[-(topk + 1):]
        ind = ind[ind != i]  # 排除自身节点
        for v in ind:
            if weight:
                sim_matrix[i][v] = sim_feat[i][v]
                sim_matrix[v][i] = sim_feat[v][i]
            else:
                sim_matrix[i][v] = 1
                sim_matrix[v][i] = 1

    # 转换为PyTorch张量
    adj_matrix_tensor = torch.tensor(sim_matrix, dtype=torch.float32)

    if loop:
        num_nodes = adj_matrix_tensor.size(0)
        eye_tensor = torch.eye(num_nodes, dtype=torch.float32)
        adj_matrix_tensor += eye_tensor

    # 进行对称化（可选，但通常在度归一化之前）
    sym_adj_matrix = (adj_matrix_tensor + adj_matrix_tensor.t()) / 2

    norm_adj_matrix = (sym_adj_matrix - torch.min(sym_adj_matrix)) / (torch.max(sym_adj_matrix) - torch.min(sym_adj_matrix))  # 5、同构图

    # # 进行度归一化
    # norm_adj_matrix = degree_normalize(sym_adj_matrix)

    return norm_adj_matrix

# ## 根据相似度阈值生成相似图
def sim_graph(feat, threshold, weight=False, loop=True):
    # 计算相似度矩阵
    sim_feat = cosine_similarity(feat)
    num_nodes = sim_feat.shape[0]
    # sim_matrix = np.zeros((num_nodes, num_nodes))
    #
    # # 遍历所有节点对，判断是否建立边
    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         if i != j and sim_feat[i][j] > threshold:
    #             if weight:
    #                 sim_matrix[i][j] = sim_feat[i][j]
    #             else:
    #                 sim_matrix[i][j] = 1

    s = torch.from_numpy(sim_feat).to(torch.float32).cuda()
    adj_matrix_tensor = torch.where(s > threshold, torch.ones_like(s), torch.zeros_like(s))

    # # 转换为PyTorch张量
    # adj_matrix_tensor = torch.tensor(sim_matrix, dtype=torch.float32)

    # if loop:
    #     eye_tensor = torch.eye(num_nodes, dtype=torch.float32)
    #     adj_matrix_tensor += eye_tensor

    # 进行对称化（可选，但通常在度归一化之前）
    sym_adj_matrix = (adj_matrix_tensor + adj_matrix_tensor.t())

    norm_adj_matrix = (sym_adj_matrix - torch.min(sym_adj_matrix)) / (torch.max(sym_adj_matrix) - torch.min(sym_adj_matrix))  # 5、同构图

    # # 进行度归一化
    # norm_adj_matrix = degree_normalize(sym_adj_matrix)

    return norm_adj_matrix


# def degree_normalize(adj_matrix):
#     # 计算度矩阵
#     degree = torch.sum(adj_matrix, dim=1)
#     degree_inv_sqrt = torch.pow(degree, -0.5)
#     degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # 防止除零错误
#
#     # 计算度归一化
#     norm_adj_matrix = adj_matrix * degree_inv_sqrt[:, None] * degree_inv_sqrt[None, :]
#     return norm_adj_matrix

def load_DBLP_data(prefix=r'.\Dataset\DBLP'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20)

    labels = np.load(prefix + '/labels.npy')

    # ## 构建特征相似图
    sm_adj = knn_graph(features_0, 300)  # ## 根据相似度节点的数量生成相似图
    # sm_adj = sim_graph(features_0, 0.31)  # ## 根据相似度阈值生成相似图

    nd_APA = scipy.sparse.load_npz(prefix + '/apa.npz').A
    nd_APTPA = scipy.sparse.load_npz(prefix + '/aptpa.npz').A
    nd_APCPA = scipy.sparse.load_npz(prefix + '/apvpa.npz').A

    nd_APA[nd_APA != 0] = 1
    nd_APTPA[nd_APTPA != 0] = 1
    nd_APCPA[nd_APCPA != 0] = 1

    APA = nd_APA
    APTPA = nd_APTPA
    APCPA = nd_APCPA

    ndarray_list = [APA, APTPA, APCPA]
    gs = [torch.tensor(arr, dtype=torch.float32) for arr in ndarray_list]

    # # ## 将元路径子图转换为稀疏DGLGraph
    # apa_g = dgl.from_scipy(sp.csr_matrix(nd_APA))
    # apcpa_g = dgl.from_scipy(sp.csr_matrix(nd_APTPA))
    # aptpa_g = dgl.from_scipy(sp.csr_matrix(nd_APCPA))
    # gs = [apa_g, apcpa_g, aptpa_g]  # 1、元路径子图

    adj = 0.8 * APA + 0.35 * APTPA + 0.25 * APCPA
    # adj = APA + APTPA + APCPA
    adj = torch.Tensor(adj)
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))  # 5、同构图

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')
    ADJ = dgl.DGLGraph(ADJ + (ADJ.T))
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    features_0 = torch.FloatTensor(features_0)  # 2、目标节点特征
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    features = [features_0, features_1, features_2, features_3]

    labels = torch.LongTensor(labels)  # 3、标签
    type_mask = np.load(prefix + '/node_types.npy')  # 节点类型的掩码

    num_classes = 4  # 标签类属

    return gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj


def load_ACM4025(prefix=r'.\Dataset\ie_ACM4025'):
    # 从 .pkl 文件读取数据
    with open(prefix + '/feature.pkl', 'rb') as f:
        features = pickle.load(f)

    labels = np.load(prefix + '/label.npy')
    features_0 = features[0].numpy()
    # ## 构建特征相似图
    sm_adj = knn_graph(features_0, 300)  # ## 根据相似度节点的数量生成相似图
    # sm_adj = sim_graph(features_0, 0.3)  # ## 根据相似度阈值生成相似图

    nd_PAP = np.load(prefix + '/pap.npy')  # 是无向图
    nd_PSP = np.load(prefix + '/psp.npy')  # 是无向图

    nd_PAP[nd_PAP != 0] = 1
    nd_PSP[nd_PSP != 0] = 1

    PAP = nd_PAP
    PSP = nd_PSP

    ndarray_list = [PAP, PSP]
    gs = [torch.tensor(arr, dtype=torch.float32) for arr in ndarray_list]

    # adj = 0.55 * PAP + 0.45 * PSP
    adj = PAP + PSP
    adj = torch.Tensor(adj)
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))  # 5、同构图

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')
    ADJ.data[:] = 1
    ADJ = dgl.DGLGraph((ADJ + (ADJ.T))/2)
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    labels = torch.LongTensor(labels)  # 3、标签
    # 创建三个一维数组，分别填充 0, 1, 2
    array1 = np.zeros(features[0].shape[0], dtype=int)  # 全部为 0
    array2 = np.ones(features[1].shape[0], dtype=int)  # 全部为 1
    array3 = np.full(features[2].shape[0], 2, dtype=int)  # 全部为 2

    # 合并这三个数组
    type_mask = np.concatenate([array1, array2, array3])
    # 节点类型的掩码

    num_classes = 3  # 标签类属

    return gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj


def load_ourACM3025(prefix=r'.\Dataset\ACM'):
    # 从 .pkl 文件读取数据
    with open(prefix + '/feature.pkl', 'rb') as f:
        features = pickle.load(f)

    labels = np.load(prefix + '/label.npy')
    features_0 = features[0].numpy()
    # ## 构建特征相似图
    # sm_adj = knn_graph(features_0, 300)  # ## 根据相似度节点的数量生成相似图
    sm_adj = sim_graph(features_0, 0.29)  # ## 根据相似度阈值生成相似图, 当设定为0.3时，平均节点数：226.2463
    # ## 计算平均边数的代码：sum(torch.sum(sm_adj != 0, dim=1))/3025
    nd_PAP = np.load(prefix + '/pap.npy')
    nd_PSP = np.load(prefix + '/psp.npy')
    nd_PSPAP = np.load(prefix + '/pspap.npy')
    nd_PAPSP = np.load(prefix + '/papsp.npy')

    nd_PAP[nd_PAP != 0] = 1
    nd_PSP[nd_PSP != 0] = 1
    nd_PSPAP[nd_PSPAP != 0] = 1
    nd_PAPSP[nd_PAPSP != 0] = 1

    PAP = nd_PAP
    PSP = nd_PSP
    PSPAP = nd_PSPAP
    PAPSP = nd_PAPSP

    ndarray_list = [PAP, PSP, PSPAP, PAPSP]
    gs = [torch.tensor(arr, dtype=torch.float32) for arr in ndarray_list]
    # adj = 0.55 * PAP + 0.45 * PSP
    adj = PAP + PSP + PSPAP + PAPSP

    adj = torch.Tensor(adj)  # 每个目标节点的平均邻居数：731.1097
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))  # 5、同构图

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')  # 异构图不是对称矩阵
    # 将所有非零值设定为 1
    ADJ.data[:] = 1
    ADJ = dgl.DGLGraph(ADJ + (ADJ.T))
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    labels = torch.LongTensor(labels)  # 3、标签
    # 创建三个一维数组，分别填充 0, 1, 2
    array1 = np.zeros(features[0].shape[0], dtype=int)  # 全部为 0
    array2 = np.ones(features[1].shape[0], dtype=int)  # 全部为 1
    array3 = np.full(features[2].shape[0], 2, dtype=int)  # 全部为 2

    # 合并这三个数组
    type_mask = np.concatenate([array1, array2, array3])
    # 节点类型的掩码

    num_classes = 3  # 标签类属

    return gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj


def load_ieIMDB3228(prefix=r'.\Dataset\ie_IMDB3228'):
    # 从 .pkl 文件读取数据
    with open(prefix + '/feature.pkl', 'rb') as f:
        features = pickle.load(f)

    labels = np.load(prefix + '/label.npy')
    features_0 = features[0].numpy()
    # ## 构建特征相似图
    sm_adj = knn_graph(features_0, 300)  # ## 根据相似度节点的数量生成相似图  # ## 当设定为300时，平均节点数：400
    # sm_adj = sim_graph(features_0, 0.3)  # ## 根据相似度阈值生成相似图
    # ## 计算平均边数的代码：sum(torch.sum(sm_adj != 0, dim=1))/3228
    nd_MAM = np.load(prefix + '/mam.npy')  # 是无向图
    nd_MDM = np.load(prefix + '/mdm.npy')  # 是无向图
    nd_MUM = np.load(prefix + '/mum.npy')  # 是无向图

    nd_MAM[nd_MAM != 0] = 1
    nd_MDM[nd_MDM != 0] = 1
    nd_MUM[nd_MUM != 0] = 1

    MAM = nd_MAM
    MDM = nd_MDM
    MUM = nd_MUM

    ndarray_list = [MAM, MDM, MUM]
    gs = [torch.tensor(arr, dtype=torch.float32) for arr in ndarray_list]

    adj = 5 * MAM + 2 * MDM + 2 * MUM
    adj = torch.Tensor(adj)  # 构建的节点数：1486
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))  # 5、同构图

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')
    ADJ.data[:] = 1
    ADJ = dgl.DGLGraph((ADJ + (ADJ.T))/2)
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    labels = torch.LongTensor(labels)  # 3、标签
    # 创建三个一维数组，分别填充 0, 1, 2
    array1 = np.zeros(features[0].shape[0], dtype=int)  # 全部为 0
    array2 = np.ones(features[1].shape[0], dtype=int)  # 全部为 1
    array3 = np.full(features[2].shape[0], 2, dtype=int)  # 全部为 2
    array4 = np.full(features[3].shape[0], 3, dtype=int)  # 全部为 2

    # 合并这三个数组
    type_mask = np.concatenate([array1, array2, array3, array4])
    # 节点类型的掩码

    num_classes = 4  # 标签类属

    return gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj


def load_YELP(prefix=r'.\Dataset\YELP'):
    # 从 .pkl 文件读取数据
    with open(prefix + '/feature.pkl', 'rb') as f:
        features = pickle.load(f)

    labels = np.load(prefix + '/label.npy')
    features_0 = features[0].numpy()
    # ## 构建特征相似图
    sm_adj = knn_graph(features_0, 300)  # ## 根据相似度节点的数量生成相似图
    # sm_adj = sim_graph(features_0, 0.3)  # ## 根据相似度阈值生成相似图


    nd_BUB = np.load(prefix + '/BUB.npy')  # 是无向图
    nd_BSB = np.load(prefix + '/BSB.npy')  # 是无向图
    nd_BUBSB = np.load(prefix + '/BUBSB.npy')  # 是无向图
    nd_BUBLB = np.load(prefix + '/BUBLB.npy')  # 是无向图
    nd_BLB = np.load(prefix + '/BLB.npy')  # 是无向图

    nd_BUB[nd_BUB != 0] = 1
    nd_BSB[nd_BSB != 0] = 1
    nd_BUBSB[nd_BUBSB != 0] = 1
    nd_BUBLB[nd_BUBLB != 0] = 1
    nd_BLB[nd_BLB != 0] = 1

    BUB = nd_BUB
    BSB = nd_BSB
    BUBSB = nd_BUBSB
    BUBLB = nd_BUBLB
    BLB = nd_BLB

    # BUB_g = dgl.from_scipy(sp.csr_matrix(BUB))
    # BSB_g = dgl.from_scipy(sp.csr_matrix(BSB))
    # BUBSB_g = dgl.from_scipy(sp.csr_matrix(BUBSB))
    # BUBLB_g = dgl.from_scipy(sp.csr_matrix(BUBLB))
    # gs = [BUB_g, BSB_g, BUBSB_g, BUBLB_g]  # 1、元路径子图

    # ndarray_list = [BUB, BSB, BUBSB, BLB]
    # gs = [torch.tensor(arr, dtype=torch.float32) for arr in ndarray_list]
    # # adj = 0.28*BUB + 0.3*BSB + 0.2*BUBSB + 0.22*BLB

    ndarray_list = [BUB, BSB, BLB, BUBSB, BUBLB]
    gs = [torch.tensor(arr, dtype=torch.float32) for arr in ndarray_list]
    adj = BUB + BSB + BLB + BUBSB + BUBLB
    # adj = 0.29*BUB + 0.27*BSB + 0.22*BUBSB + 0.22*BUBLB

    adj = torch.Tensor(adj)
    adj = (adj - torch.min(adj)) / (torch.max(adj) - torch.min(adj))  # 5、同构图

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz')
    ADJ.data[:] = 1
    ADJ = dgl.DGLGraph((ADJ + (ADJ.T))/2)
    ADJ = dgl.remove_self_loop(ADJ)
    ADJ = dgl.add_self_loop(ADJ)

    labels = torch.LongTensor(labels)  # 3、标签
    # 创建三个一维数组，分别填充 0, 1, 2
    array1 = np.zeros(features[0].shape[0], dtype=int)  # 全部为 0
    array2 = np.ones(features[1].shape[0], dtype=int)  # 全部为 1
    array3 = np.full(features[2].shape[0], 2, dtype=int)  # 全部为 2
    array4 = np.full(features[3].shape[0], 3, dtype=int)  # 全部为 2

    # 合并这三个数组
    type_mask = np.concatenate([array1, array2, array3, array4])
    # 节点类型的掩码


    num_classes = 3  # 标签类属

    return gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj

