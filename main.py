import torch
# from utils import EarlyStopping, load_data
from utils import EarlyStopping
from data import load_ourdata
from evaluate import metric_sets
import numpy as np
from sklearn import metrics
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from data import load_ACM_data, load_IMDB_data, load_DBLP_data, load_ACM4025, load_ourACM3025, load_ieIMDB3228, load_YELP

# # 消除FutureWarning
# from warnings import simplefilter
# simplefilter(action="ignore", category=FutureWarning)

def main(args):
    # result storage
    path = Path(args["log_dir"])
    writer = SummaryWriter(path)

    # para1 = args["α"]  # 结构重构损失前面的系数
    # para2 = args["β"]  # 特征重构损失前面的系数

    para1 = args["α"]  # 结构重构损失前面的系数
    para2 = args["b"]  # 特征重构损失前面的系数
    para3 = args["c"]  # 模块度前面的系数

    # (
    #     gs,
    #     features,
    #     labels,
    #     num_classes,
    #     adj,
    # ) = load_data(args["dataset"])
    # gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj = load_DBLP_data()
    # gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj = load_ACM_data()
    # gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj = load_ACM4025()
    # gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj = load_ourACM3025()
    # gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj = load_ieIMDB3228()
    # gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj = load_YELP()

    gs, features, labels, num_classes, adj, ADJ, type_mask, sm_adj = load_ourdata(args["dataset"])

    # 加载数据集，分别返回元路径子图列表、节点特征、标签、标签类别数、重建的同构图, 异构原图，节点类型掩码
    in_dims = [feature.shape[1] for feature in features]  # 输入每种类型节点的属性维度
    features = [feat.to(args['device']) for feat in features]  # # 特征迁移到GPU上
    feature = features[0]  # 取出目标节点
    ADJ = ADJ.to(args['device'])  # 原始的异构图邻接矩阵
    norm = [1, 0]
    adj = adj.to(args["device"])  # 重构后的标准同构图迁移到GPU上
    sm_adj = sm_adj.to(args["device"])
    # labels = labels.to(args["device"])

    from model import HAESF  # 加载模型文件
    model = HAESF(
        num_meta_paths=len(gs),
        node_size=feature.shape[0],
        in_size=in_dims,
        hidden_size=args["hidden_units"],
        out_size=num_classes,
        dropout=args["dropout"],
        ho_adj=adj,
    ).to(args["device"])  # 输入分别是元路径子图数，节点数量，输入节点特征维度，设置隐藏层维度，分类数，注意力头数，

    g = [graph.to(args["device"]) for graph in gs]  # 遍历元路径子图列表转到显卡上

    # 根据情况灵活选择NMF变体
    # single-layer NMF
    # U = torch.rand(features.shape[0], num_classes)
    # V = torch.Tensor(np.random.random([num_classes, features.shape[1]]))

    # # double-layer NMF
    # W1 = torch.rand(feature.shape[0], args["hidden_units"])
    # W2 = torch.rand(args["hidden_units"], num_classes)
    # S = torch.rand(num_classes, feature.shape[1])
    #
    # # # # NMF layer
    # # model.cluster_layer1.data = U.to(args['device'])
    # # model.cluster_layer2.data = V.to(args['device'])
    # model.cluster_layer1.data = W1.to(args['device'])
    # model.cluster_layer2.data = W2.to(args['device'])
    # model.cluster_layer3.data = S.to(args['device'])

    # best_loss
    stopper = EarlyStopping(patience=args["patience"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    print("Start training")
    best_nmf_loss = 188.0
    best_modu_loss = 0.0
    best_feat_loss = 100.0
    for epoch in tqdm(range(args["num_epochs"])):
        model.train()
        hid, embedding, adj_pred, V, emb = model(features, g, ADJ, type_mask, norm, args['agg'], sm_adj)  # 输入：元路径子图列表，特征矩阵
        import torch.nn.functional as F
        # feat_loss = F.mse_loss(feature, V)  # 重构属性损失，均方差损失适用于一方数据为0，1二元分布，另一方为概率分布
        feat_loss = torch.nn.functional.binary_cross_entropy(V, feature)  # 计算属性重构损失，使用二元交叉熵损失
        modu_loss = torch.trace(emb.t() @ model.B @ emb)  # modularity，模块化（计算对角线元素和）([7,2708]*[2708,2708]*[2708,7]=[7,7])
        # # 输出为节点的多注意力头维的嵌入(N, D * K)，预测标签概率，重构同构图，S:(类别数，属性维度)
        # nmf_loss = torch.sqrt(torch.norm(feature - model.cluster_layer1 @  model.cluster_layer2 @ V))
        # # 计算原始特征矩阵与S重构特征矩阵之间的差异。 公式(13)
        # unity_loss = torch.sqrt(torch.norm(feature - embedding @ V))  # 计算原始特征矩阵与由嵌入生成的特征矩阵之间的差异。公式(15)
        #
        # # re_loss = F.mse_loss(adj_pred, adj)
        re_loss = torch.nn.functional.binary_cross_entropy(adj_pred, adj)  # 计算结构重构损失，使用二元交叉熵损失

        loss = para1 * re_loss + para2 * feat_loss + - para3 * modu_loss
        # print(f" 结构损失：{re_loss}，属性损失：{feat_loss}， 模块度损失：{modu_loss}，###， 总体损失：{loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, indices = torch.max(embedding, dim=1)
        prediction = indices.long().cpu().numpy()
        NMI = metrics.normalized_mutual_info_score(prediction, labels)

        early_stop = stopper.step(loss.data.item(), NMI, model, epoch)

        # print("Epoch {:d} | Train Loss {:.4f} | NMI {}".format(epoch + 1, loss.item(), NMI,))
        with open(path / 'result.txt', 'a') as f:
            f.write("Epoch {:d} | Train Loss {:.4f} | NMI {}".format(epoch + 1, loss.item(), NMI,))
            f.write(f'\n')
        if early_stop:
            break

    result = metric_sets(model, g, num_classes, features, labels.tolist(), ADJ, type_mask, norm, args['agg'], sm_adj, adj)

    print("Final result_Kmeans :NMI {} | Ari {:.4f}| Acc {:.4f} | F1_macro {:.4f}".format(
        result[0], result[1], result[2], result[3]))
    # print(f"另外测试模块度与属性熵的数值为：Q={result[4]};  |  AE={result[5]}")

    # stopper.load_checkpoint(model)
    # 将每个元素取小数点后四位
    rounded_list = [round(num, 4) for num in result]
    return rounded_list

if __name__ == "__main__":
    import argparse
    from data import setup

    parser = argparse.ArgumentParser("HAE-SF")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )

    # The default configuration
    default_configure = {
        "lr": 0.005,  # 原来是： 0.005
        # "num_heads": [8],
        "hidden_units": 64,
        "dropout": 0.35,  # HANSF中为0.6， OSGNN中为0.5
        "weight_decay": 0.001,
        "num_epochs": 3,
        "patience": 100,
        "seed": 6,
        "α": 50,
        # "β": 0.25,
        "b": 0.25,
        "c": 0.00008,
        'agg': 'att'  # 'mean'，'contact'，'att'
    }

    args = parser.parse_args().__dict__
    args.update(default_configure)
    args = setup(args)
    main(args)

    # NMI = []
    # Ari = []
    # Acc = []
    # F1_macro = []
    # import datetime
    # import json
    # dt = datetime.datetime.now()
    # result_file = "results/{}_{}_{:02d}-{:02d}-{:02d}.pth".format(
    #     args['dataset'], dt.date(), dt.hour, dt.minute, dt.second
    # )
    # for i in range(10):
    #     result = main(args)
    #     NMI.append(result[0])
    #     Ari.append(result[1])
    #     Acc.append(result[2])
    #     F1_macro.append(result[3])
    # print(f"测试数据：{args['dataset']}\n",
    #       f"NMI测评:{NMI}\n",
    #       f"Ari测评:{Ari}\n",
    #       f"Acc测评:{Acc}\n",
    #       f"F1_macro测评:{F1_macro}\n")
    # NMI_ave = np.mean(NMI)
    # Ari_ave = np.mean(Ari)
    # Acc_ave = np.mean(Acc)
    # F1_macro_ave = np.mean(F1_macro)
    # print(f"测试平均数据：{args['dataset']}",
    #       f"NMI平均测评:{NMI_ave}",
    #       f"Ari平均测评:{Ari_ave}",
    #       f"Acc平均测评:{Acc_ave}",
    #       f"F1_macro平均测评:{F1_macro_ave}")
    # result_dic = {
    #       "NMI测评": NMI,
    #       "Ari测评": Ari,
    #       "Acc测评": Acc,
    #       "F1_macro测评": F1_macro}
    # # 将字典保存为 JSON 文件
    # with open(result_file, 'w') as json_file:
    #     json.dump(result_dic, json_file)
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()