import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import random
import torch


def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def find_one_zero(RS):
    one_postive = np.argwhere(RS == 1)
    one_postive = np.array(one_postive)
    np.random.shuffle(one_postive)

    zero_postive = np.argwhere(RS == 0)
    zero_postive = np.array(zero_postive)
    np.random.shuffle(zero_postive)  

    return one_postive, zero_postive

def get_node_types(mirna_disease_matrix):
    num_miRNA = mirna_disease_matrix.shape[0]
    num_disease = mirna_disease_matrix.shape[1]

    node_types = torch.tensor(
        [0] * num_miRNA +   # miRNA
        [1] * num_disease   # disease
    )
    return node_types

def index_dp(A, k):
    index_drug_p = np.transpose(np.nonzero(A))
    np.random.shuffle(index_drug_p)
    data_1 = np.array_split(index_drug_p, k, 0)

    index_dp_zero = np.argwhere(A == 0)
    np.random.shuffle(index_dp_zero)
    data_0 = np.array_split(index_dp_zero, k, 0)

    return data_0, data_1

def Split_data(data_1, data_0, fold, k, drug_p):
    X_train = []  
    X_test = []

    for i in range(k):  
        if i != fold:  # 如果不是当前折，则用作训练集
            for j in range(len(data_1[i])):
                X_train.append(data_1[i][j])
            for t in range(len(data_0[i])):
                if t < len(data_1[i]):  # 平衡正负样本数量
                    X_train.append(data_0[i][t])
                else:
                    x = int(data_0[i][t][0])
                    y = int(data_0[i][t][1])
                    X_test.append([x, y])
        else:  # 当前折用作测试集
            for t1 in range(len(data_1[i])):
                x = int(data_1[i][t1][0])  
                y = int(data_1[i][t1][1])  
                X_test.append([x, y])
                    
            for t2 in range(len(data_0[i])):  
                x = int(data_0[i][t2][0])  
                y = int(data_0[i][t2][1])
                X_test.append([x, y])

    np.random.shuffle(X_train)
    return X_train, X_test

def Preproces_Data(RS, test_id):
    copy_RS = RS / 1
    for i in range(test_id.shape[0]):
        x = int(test_id[i][0])
        y = int(test_id[i][1])
        copy_RS[x, y] = 0
    return copy_RS

def load_data(id, RS):
    import torch.utils.data as Data
    x = []
    y = []
    for j in range(id.shape[0]):  
        temp_save = []
        x_A = int(id[j][0])  
        y_A = int(id[j][1])  
        temp_save.append([x_A, y_A])  
        
        label = RS[x_A, y_A]
        
        x.append(temp_save)
        y.append(label)

    x = torch.FloatTensor(np.array(x))   # shape: (N, 1, 2)
    y = torch.LongTensor(np.array(y))   # shape: (N,)

    torch_dataset = Data.TensorDataset(x, y)
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=len(x),     # 一次加载所有数据
        shuffle=False,
        num_workers=0,
        drop_last=False
    )  
    return x, y, data2_loader
def constructNet(mirna_disease_matrix):
    drug_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[0], mirna_disease_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[1], mirna_disease_matrix.shape[1]), dtype=np.int8))
    matrix1 = mirna_disease_matrix*mirna_disease_matrix.T
    matrix2 = mirna_disease_matrix.T*mirna_disease_matrix

    mat1 = np.hstack((matrix1, mirna_disease_matrix))#沿着水平方向进行连接
    mat2 = np.hstack((mirna_disease_matrix.T, matrix2))
    adj = np.vstack((mat1, mat2))#沿着垂直方向进行连接
    return adj

def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix):
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix*0.3))
    mat2 = np.hstack((0.3*train_mirna_disease_matrix.T, disease_matrix))
    node_embeddings = np.vstack((mat1, mat2))
    return node_embeddings


def gaussian_interaction_profile_similarity(profile_matrix):
    profile_matrix = np.asarray(profile_matrix, dtype=np.float32)
    squared_norm = np.sum(profile_matrix * profile_matrix, axis=1)
    gamma = 1.0 / (np.mean(squared_norm) + 1e-8)
    diff = profile_matrix[:, None, :] - profile_matrix[None, :, :]
    distance = np.sum(diff * diff, axis=-1)
    sim = np.exp(-gamma * distance)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def fuse_similarity(base_similarity, gip_similarity, gip_weight):
    base_similarity = np.asarray(base_similarity, dtype=np.float32)
    gip_similarity = np.asarray(gip_similarity, dtype=np.float32)
    weight = float(gip_weight)
    fused = (1.0 - weight) * base_similarity + weight * gip_similarity
    fused = 0.5 * (fused + fused.T)
    return fused.astype(np.float32)

def train_features_choose(rel_adj_mat, features_embedding, device, forbidden_indices=None):
    
    if not isinstance(rel_adj_mat, torch.Tensor):
        rel_adj_mat = torch.from_numpy(rel_adj_mat).to(features_embedding.device).float()  # 确保在 GPU 上
    rna_nums = rel_adj_mat.shape[0]
    features_embedding_rna = features_embedding[:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:, :]
    
    # 向量化正样本
    positive_indices = torch.nonzero(rel_adj_mat == 1, as_tuple=False)  # [num_pos, 2]
    pos_r = positive_indices[:, 0]
    pos_d = positive_indices[:, 1]
    pos_features = features_embedding_rna[pos_r] * features_embedding_dis[pos_d]  # [num_pos, dim]
    
    valid_negative = rel_adj_mat == 0
    if forbidden_indices is not None:
        forbidden_indices = torch.as_tensor(
            forbidden_indices,
            dtype=torch.long,
            device=features_embedding.device,
        ).view(-1, 2)
        if forbidden_indices.numel() > 0:
            valid_rows = forbidden_indices[:, 0].clamp(0, rel_adj_mat.shape[0] - 1)
            valid_cols = forbidden_indices[:, 1].clamp(0, rel_adj_mat.shape[1] - 1)
            valid_negative[valid_rows, valid_cols] = False

    # 生成负样本：为每个正样本随机选1个负，并排除当前验证fold中的pair
    num_pos = positive_indices.shape[0]
    neg_d = []
    for row in pos_r.tolist():
        candidates = torch.where(valid_negative[row])[0]
        if candidates.numel() == 0:
            raise ValueError(f"No valid negative samples remain for row {row}.")
        selected = candidates[torch.randint(0, candidates.numel(), (1,), device=features_embedding.device)]
        neg_d.append(selected)
    neg_d = torch.cat(neg_d, dim=0)
    
    neg_features = features_embedding_rna[pos_r] * features_embedding_dis[neg_d]  # [num_pos, dim]
    
    # 合并特征和标签
    train_features_input = torch.cat([pos_features, neg_features], dim=0)  # [2*num_pos, dim]
    train_label = torch.cat([torch.ones(num_pos, 1), torch.zeros(num_pos, 1)], dim=0).to(features_embedding.device)
    
    return train_features_input, train_label




def test_features_choose(rel_adj_mat, features_embedding, device):
    if not isinstance(rel_adj_mat, torch.Tensor):
        rel_adj_mat = torch.from_numpy(rel_adj_mat).to(features_embedding.device).float()
    rna_nums, dis_nums = rel_adj_mat.shape
    features_embedding_rna = features_embedding[:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:, :]
    
    # 向量化：广播乘法
    test_features_input = features_embedding_rna[:, None, :] * features_embedding_dis[None, :, :]  # [rna, dis, dim]
    test_features_input = test_features_input.view(rna_nums * dis_nums, -1)  # [rna*dis, dim]
    
    # 标签：直接展平 rel_adj_mat
    test_label = rel_adj_mat.view(-1, 1)  # [rna*dis, 1]
    
    return test_features_input, test_label

def sort_matrix(score_matrix, interact_matrix):
    '''
    实现矩阵的列元素从大到小排序
    1、np.argsort(data,axis=0)表示按列从小到大排序
    2、np.argsort(data,axis=1)表示按行从小到大排序
    '''
    sort_index = np.argsort(-score_matrix, axis=0)  # 沿着行向下(每列)的元素进行排序
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted


def adjacency_matrix_to_index(adjacency_matrix):
  adjacency_matrix = torch.tensor(adjacency_matrix)
  num_nodes = adjacency_matrix.shape[0]
  index = torch.nonzero(adjacency_matrix, as_tuple=False).t()
  return index


#计算邻接矩阵
def get_adjacency_matrix(similarity_matrix, threshold):
    n = similarity_matrix.shape[0]
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] >= threshold:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix


def matrix_to_edge_index(H, n_m, n_d,
                             thr_mm=None, thr_dd=None, thr_md=None,
                             keep_self=False, sym_mm=True, sym_dd=True):
    """
    纯阈值稀疏化版本：
    - MM/DD/MD 分别用 thr_mm/thr_dd/thr_md 过滤
    - 保留 B > thr 的边
    返回 edge_index(2,E), edge_weight(E,)
    """
    H = torch.as_tensor(H).float()
    device = H.device
    N = n_m + n_d

    G = H.clone()
    if not keep_self:
        G.fill_diagonal_(0.0)

    # 分块
    MM = G[:n_m, :n_m]
    MD = G[:n_m, n_m:]
    DD = G[n_m:, n_m:]

    # 相似块对称化
    if sym_mm:
        MM = 0.5 * (MM + MM.t())
    if sym_dd:
        DD = 0.5 * (DD + DD.t())

    def _thr(B, thr):
        if thr is None:
            return B
        thr = float(thr)
        return torch.where(B > thr, B, torch.zeros_like(B))

    MM = _thr(MM, thr_mm)
    DD = _thr(DD, thr_dd)
    MD = _thr(MD, thr_md)
    DM = MD.t()  # 强制 MD/DM 一致（如果你要保留原 DM，就自己单独算 DM）

    # 拼回去
    G2 = torch.zeros((N, N), device=device, dtype=H.dtype)
    G2[:n_m, :n_m] = MM
    G2[:n_m, n_m:] = MD
    G2[n_m:, :n_m] = DM
    G2[n_m:, n_m:] = DD

    edge_index = torch.nonzero(G2, as_tuple=False).t().contiguous()
    src, dst = edge_index[0], edge_index[1]
    edge_weight = G2[src, dst].contiguous()
    return edge_index, edge_weight



def transfer_label_from_prob(proba, threshold):
    proba = (proba - proba.min()) / (
            proba.max() - proba.min())
    label = [1 if val >= threshold else 0 for val in proba]
    return label


def balance_samples(pos_idx, neg_idx, neg_pos_ratio=1):
    pos_num = len(pos_idx)
    neg_num = min(len(neg_idx), pos_num * neg_pos_ratio)

    neg_idx = random.sample(neg_idx, neg_num)
    return pos_idx, neg_idx

def random_index(index_matrix, sizes):
    set_seed(sizes.seed)
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)  # 获得相同随机数
    random.shuffle(random_index)  # 将原列表的次序打乱
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp

class Sizes(object):
    def __init__(self, drug_size, mic_size):
        self.c = 12

'''def get_adjacency_matrix(feat, k):
    # compute C
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C'''

def plot_auc_curves(fprs, tprs, aucs):
    mean_fpr = np.linspace(0, 1, 1000)
    tpr = []
    #plt.style.use('ggplot')
    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.8, label='ROC fold %d (AUC = %.4f)' % (i + 1, aucs[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    auc_std = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))
    plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle='--', color='navy', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc='lower right', prop={"size": 8})
    #plt.savefig('./auc.jpg', dpi=1200, bbox_inches='tight') 
    plt.show()
    

def plot_prc_curves(precisions, recalls, auprs):
    mean_recall = np.linspace(0, 1, 1000)
    precision = []
    #plt.style.use('ggplot')
    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.8, label='ROC fold %d (AUPR = %.4f)' % (i + 1, auprs[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(auprs)
    prc_std = np.std(auprs)
    plt.plot(mean_recall, mean_precision, color='b', alpha=0.8,
             label='Mean AUPR (AUPR = %.4f $\pm$ %.4f)' % (mean_prc, prc_std))  # AP: Average Precision
    plt.plot([-0.05, 1.05], [1.05, -0.05], linestyle='--', color='navy', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc='lower left', prop={"size": 8})
    #plt.savefig('./pr.jpg', dpi=1200, bbox_inches='tight') 
    plt.show()
