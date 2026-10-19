import numpy as np
import torch
from clac_metric import get_metrics
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from utils import *
from model import *
import gc
import argparse
import torch.nn.functional as F
import warnings
import configparser
import os
from contextlib import nullcontext
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = 'MDAD'

bootstrap_parser = argparse.ArgumentParser(add_help=False)
bootstrap_parser.add_argument('--dataset', default=DEFAULT_DATASET, choices=['MDAD', 'aBiofilm'])
bootstrap_args, _ = bootstrap_parser.parse_known_args()
dataset = bootstrap_args.dataset
ini_path = PROJECT_ROOT / 'config' / f'{dataset}.ini'

# 读取配置
config = configparser.ConfigParser()
config.optionxform = str  
config.read(ini_path)
dataset_config = config[dataset]


def parse_config_value(value):
    value = value.strip()
    if ',' in value:
        return value
    try:
        if any(mark in value for mark in ('.', 'e', 'E')):
            return float(value)
        return int(value)
    except ValueError:
        return value


parser = argparse.ArgumentParser(parents=[bootstrap_parser])
for key, value in dataset_config.items():
    value = parse_config_value(value)
    parser.add_argument(f'--{key}', default=value, type=type(value))
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置PyTorch内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def clear_device_cache():
    if device.type == "cuda":
        torch.cuda.empty_cache()


def autocast_context(enabled):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def normalize_score_matrix(score_matrix):
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    score_min = np.min(score_matrix)
    score_max = np.max(score_matrix)
    return (score_matrix - score_min) / (score_max - score_min + 1e-8)

def cross_validation_experiment(args):
    fold_dir = PROJECT_ROOT / 'dataset' / args.dataset / 'cv_folds' / f"seed_{int(args.seed)}_k_{int(args.k_fold)}"
    print(f"Using self-contained CV folds: {fold_dir}")
    metric = np.zeros((1, 7))
    score =[]
    tprs=[]
    fprs=[]
    aucs=[]
    precisions=[]
    recalls = []
    auprs = []
    metrics_all = [] 
    out_all=[]
    strategy_weights_all=[]
    spectral_bias_all = []
    pre_matrix = None
    print("seed=%d, evaluating drug-microbe...." % (args.seed))
    # 启用混合精度训练
    use_amp = device.type == "cuda" and bool(args.use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for k in range(args.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        # 清理GPU缓存
        clear_device_cache()
        gc.collect()
        with np.load(fold_dir / f'fold_{k + 1}.npz') as fold_data:
            heldout_index = np.asarray(fold_data["test_index"], dtype=np.int64)
            test_label = np.asarray(fold_data["test_label"])
            train_matrix = np.matrix(fold_data["train_matrix"], copy=True)
            row_simi = np.asarray(fold_data["row_simi"])
            col_simi = np.asarray(fold_data["col_simi"])
        if pre_matrix is None:
            pre_matrix = np.zeros(train_matrix.shape)
        if len(heldout_index) != len(test_label):
            raise ValueError(f"Fold {k + 1}: test_index and test_label lengths differ")
        if np.any(np.asarray(train_matrix)[heldout_index[:, 0], heldout_index[:, 1]] != 0):
            raise ValueError(f"Fold {k + 1}: held-out associations are present in train_matrix")
        x=constructHNet(train_matrix, row_simi, col_simi)
        edge_index = None
        node_types = get_node_types(train_matrix)
        microbe_len, dis_len = train_matrix.shape
        num_nodes = x.shape[0]
        
        x_cpu = torch.from_numpy(np.asarray(x)).to(torch.float32)
        x = x_cpu.to(device, non_blocking=True)
        L_normalized = None
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        del x_cpu
        gc.collect()

        ensemble_size = max(int(getattr(args, "ensemble_size", 1)), 1)
        seed_stride = int(getattr(args, "ensemble_seed_stride", 100))
        proba_sum = torch.zeros(microbe_len * dis_len, 1, dtype=torch.float32)

        for ensemble_id in range(ensemble_size):
            model_args = argparse.Namespace(**vars(args))
            model_args.seed = int(args.seed) + ensemble_id * seed_stride
            if ensemble_size > 1:
                print(f"Ensemble {ensemble_id + 1}/{ensemble_size} | seed={model_args.seed}")
            set_seed(model_args.seed)
            model = hu(model_args, device).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            #训练循环
            model.train()
            for epoch in range(args.epoch):
                optimizer.zero_grad()
                with autocast_context(use_amp):
                    pred, label = model(model_args, x, train_matrix, row_simi, col_simi, heldout_index, L_normalized, train_model=True)
                base_loss = F.binary_cross_entropy_with_logits(pred.float(), label.float())
                aux_loss = model.auxiliary_loss().float()
                loss = base_loss + aux_loss
                # 使用scaler进行反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                log_interval = max(int(getattr(args, "log_interval", 1)), 1)
                should_log = epoch == 0 or (epoch + 1) % log_interval == 0 or epoch + 1 == args.epoch
                if should_log and (ensemble_size == 1 or ensemble_id == ensemble_size - 1):
                    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | BCE: {base_loss.item():.4f} | Aux: {aux_loss.item():.4f}")
                
            # 定期清理内存
                if (epoch + 1) % 10 == 0:
                    clear_device_cache()

            # 模型评估
            model.eval()
            with torch.no_grad():
                with autocast_context(use_amp):
                    pred_scores, _ = model(model_args, x, train_matrix, row_simi, col_simi, heldout_index, L_normalized, train_model=False)
            proba_sum += torch.sigmoid(pred_scores).detach().cpu().float()
            del model, optimizer
            clear_device_cache()
            gc.collect()

        micro_dis_res = proba_sum / ensemble_size
        predict_y_proba = micro_dis_res.reshape(microbe_len, dis_len).detach().numpy()
        blend_alpha = float(getattr(args, "score_blend_alpha", 0.0))
        if blend_alpha > 0:
            topology_prior = np.asarray(row_simi) @ np.asarray(train_matrix, dtype=np.float32) @ np.asarray(col_simi)
            topology_prior = normalize_score_matrix(topology_prior)
            predict_y_proba = (1.0 - blend_alpha) * predict_y_proba + blend_alpha * topology_prior
        pre_matrix[tuple(np.array(heldout_index).T)] = predict_y_proba[tuple(np.array(heldout_index).T)]  
        test_scores = predict_y_proba[tuple(heldout_index.T)]
        metric_tmp = get_metrics(test_label, test_scores)
        fpr, tpr, t = roc_curve(test_label, test_scores)
        precision, recall, _ = precision_recall_curve(test_label, test_scores)
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
 
        auprs.append(metric_tmp[1])
        print(metric_tmp)

        metric += metric_tmp      
        score.append(pre_matrix.copy())
        clear_device_cache()
        gc.collect()
    print('Mean:', metric / args.k_fold)
    metrics_all = np.array(metrics_all)   # shape = (k,7)
    metric = np.array(metric / args.k_fold)   
    return metric, score 
        

def main(args):
    result, score = cross_validation_experiment(args)


if __name__== '__main__':
    main(args)


 
