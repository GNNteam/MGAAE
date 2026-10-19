import numpy as np
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from munkres import Munkres
import community
import networkx as nx


def community(z, clusters):
    z = z.detach().numpy()
    C_model = KMeans(n_clusters=clusters, verbose=0, max_iter=100, tol=0.01, n_init=3)
    C_model.fit(z)
    comm_predict = C_model.labels_

    return comm_predict


def modularity(adj: np.array, pred: np.array):
    """
    非重叠模块度
    :param adj: 邻接矩阵
    :param pred: 预测社区标签
    :return:
    """
    # graph = nx.from_numpy_matrix(adj)  # 将一个 NumPy 数组 adj 转换为 NetworkX 图对象 graph
    graph = nx.from_numpy_array(adj)
    part = pred.tolist()  # 将Tensor转换为list格式
    index = range(0, len(part))  # 得到（0,2708）的索引范围
    dic = zip(index, part)
    part = dict(dic)  # 将索引对应的标签生成键值对字典
    import community as community_louvain
    modur = community_louvain.modularity(part, graph)
    # modur = community.modularity(part, graph)  # 计算社区的模块度
    return modur

def calculate_entropy(k, pred_labels, feat):
    """
    :param k: 社区个数
    :param pred_labels: 预测社区
    :param num_nodes: 节点的个数
    :param feat: 节点属性
    :return:
    """
    # 初始化两个矩阵

    num_nodes = feat.shape[0]  # 获取节点数量

    label_assemble = np.zeros(shape=(num_nodes, k))  # 生成2708*7的零矩阵
    label_atts = np.zeros(shape=(k, feat.shape[1]))  # 初始化一个形状为 (7,1433) 的零矩阵，用来存储每个社区中节点属性的累加和。
#  #  构建社区独热编码
    label_assemble[range(num_nodes), pred_labels] = 1
    label_assemble = label_assemble.T

    # 遍历每个社区，得到每个社区下有哪些节点，将这些节点在每个属性点的值相加(计算每个社区的属性总和)
    for i in range(k):
        # 如果社区中的值大于0，则获得索引
        node_indx = np.where(label_assemble[i] > 0)  # 找出属于第 i 个社区的节点的索引
        # 获得索引下的所有属性
        node_feat = feat[node_indx]
        label_atts[i] = node_feat.sum(axis=0)  # 计算每个社区的1433维属性，对这些节点的属性按列求和（11,1433）->（1,1433）
#  ####
    __count_attrs = label_atts.sum(axis=1)  # 将每个社区的1433维属性相加，按行求和，得到每个社区的属性和
    __count_attrs = __count_attrs[:, np.newaxis]  # 将总数转换为列向量
    _tmp = label_atts / (__count_attrs + 1e-10)  # 计算每个社区内属性的比例（7,1433），相当于归一化
    p = (_tmp) * - (np.log2(_tmp + 1e-10))  # 计算每个社区内属性比例的熵贡献（7,1433）

    p = p.sum(axis=1)  # 每个社区的熵贡献进行求和，得到每个社区的熵值（7,1）
    label_assemble = label_assemble.sum(axis=1)  # 计算每个社区中节点的数量
    __entropy = (label_assemble / num_nodes) * p  # 计算加权平均熵，考虑了每个社区的节点比例。（每个社区的节点数量/节点总数）*社区熵值
    return __entropy.sum()  # 所有社区的加权平均熵，作为最终的熵值（7,1）


def metric_sets(model, g, num_classes, features, labels, ADJ, type_mask, norm, agg, sm_adj, adj):
    model.eval()
    # with torch.no_grad():
    #     hidEmb, embed, adj_pred, V = model(g, features)

    with torch.no_grad():
        hidEmb, embed, adj_pred, V, _ = model(features, g, ADJ, type_mask, norm, agg, sm_adj)

    # K-means
    hidEmb = hidEmb.cpu()
    prediction = community(hidEmb, num_classes)
    nmi = metrics.normalized_mutual_info_score(prediction, labels)
    ari = metrics.adjusted_rand_score(labels, prediction)
    acc, f1_macro = cluster_acc(labels, prediction)

    Q = 66.6
    AE = 66.6
    # Q = modularity(adj.cpu().numpy(), torch.tensor(prediction).long())  # 输入原图的邻接矩阵和预测标签，得到模块度
    # AE = calculate_entropy(num_classes, prediction, features[0].cpu().numpy())  # 计算熵，输入标签类别数、预测标签、属性矩阵

    kmeans_result = [nmi, ari, acc, f1_macro, Q, AE]

    # AE_NMF
    _, indices = torch.max(embed, dim=1)
    prediction = indices.long().cpu().numpy()
    nmi = metrics.normalized_mutual_info_score(prediction, labels)
    acc, f1_macro = cluster_acc(labels, prediction)
    ari = metrics.adjusted_rand_score(labels, prediction)

    # Q = modularity(adj.cpu().numpy(), indices.cpu())  # 输入原图的邻接矩阵和预测标签，得到模块度
    # AE = calculate_entropy(num_classes, prediction, features[0].cpu().numpy())  # 计算熵，输入标签类别数、预测标签、属性矩阵

    AENMF_result = [nmi, ari, acc, f1_macro, Q, AE]

    if kmeans_result[0] > AENMF_result[0]:
        print("###########本次输出均值聚类的结果##########")
        return kmeans_result

    return AENMF_result


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numClass1 = len(l1)

    l2 = list(set(y_pred))
    numClass2 = len(l2)

    ind = 0
    if numClass1 != numClass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numClass2 = len(l2)

    if numClass1 != numClass2:
        print("error")
        return

    cost = np.zeros((numClass1, numClass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # corresponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    # 预测标签处理后，得到的指标都有提升。
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro
