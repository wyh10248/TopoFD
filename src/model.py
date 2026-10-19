import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import test_features_choose, train_features_choose


def _get_arg(args, name, default):
    return getattr(args, name, default)


def _parse_float_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_01(matrix, eps=1e-8):
    matrix = matrix.float()
    mn = matrix.min()
    mx = matrix.max()
    return (matrix - mn) / (mx - mn + eps)


def _standardize(vector, eps=1e-8):
    vector = vector.float()
    return (vector - vector.mean()) / (vector.std(unbiased=False) + eps)


class MonotoneBiasTransform(nn.Module):
    """Learnable monotone map from diffusion intensity to attention bias."""

    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(1.0))
        self.raw_sharpness = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, kernels):
        scale = F.softplus(self.raw_scale) + 1e-6
        sharpness = F.softplus(self.raw_sharpness) + 1e-6
        return scale * torch.tanh(sharpness * kernels) + self.shift


class PairMLP(nn.Module):
    def __init__(self, input_size, drop_rate):
        super().__init__()
        hidden_1 = max(input_size // 2, 64)
        hidden_2 = max(input_size // 4, 32)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_1, hidden_2),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_2, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, pair_embedding):
        return self.net(pair_embedding)


class FractionalRelationAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_relations, num_heads, dropout, relation_fusion="gate"):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.relation_fusion = relation_fusion
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.out_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])

        if relation_fusion == "gate":
            self.gate_norm = nn.LayerNorm(hidden_dim * num_relations)
            self.gate_linear = nn.Linear(hidden_dim * num_relations, num_relations)

        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _split_heads(self, tensor):
        num_nodes = tensor.size(0)
        return tensor.view(num_nodes, self.num_heads, self.head_dim)

    def forward(self, x, relation_bias, relation_masks):
        relation_outputs = []
        for rel_id in range(self.num_relations):
            q = self._split_heads(self.q_proj[rel_id](x))
            k = self._split_heads(self.k_proj[rel_id](x))
            v = self._split_heads(self.v_proj[rel_id](x))

            scores = torch.einsum("nhd,mhd->hnm", q, k) / math.sqrt(self.head_dim)
            scores = scores + relation_bias[rel_id].unsqueeze(0).to(dtype=scores.dtype)

            mask = relation_masks[rel_id].unsqueeze(0)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out = torch.einsum("hnm,mhd->nhd", attn, v).reshape(x.size(0), self.hidden_dim)
            relation_outputs.append(self.out_proj[rel_id](out))

        stacked = torch.stack(relation_outputs, dim=1)
        if self.relation_fusion == "gate":
            gate_input = self.gate_norm(torch.cat(relation_outputs, dim=-1))
            gates = torch.softmax(self.gate_linear(gate_input), dim=-1)
        else:
            gates = torch.full(
                (x.size(0), self.num_relations),
                1.0 / self.num_relations,
                dtype=stacked.dtype,
                device=stacked.device,
            )
        fused = torch.sum(stacked * gates.unsqueeze(-1), dim=1)

        x = self.norm_attn(x + self.dropout(fused))
        x = self.norm_ffn(x + self.dropout(self.ffn(x)))
        return x, gates


class hu(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.num_relations = 3
        self.orders = _parse_float_list(_get_arg(args, "fd_orders", "0.25,0.5,1.0"), [0.25, 0.5, 1.0])
        self.times = _parse_float_list(_get_arg(args, "fd_times", "0.5,1.0,2.0"), [0.5, 1.0, 2.0])
        self.num_scales = len(self.orders) * len(self.times)

        self.hidden_dim = int(_get_arg(args, "hidden_dim", 128))
        self.output_dim = int(_get_arg(args, "out_dim", self.hidden_dim))
        self.num_heads = int(_get_arg(args, "num_heads", 4))
        self.num_layers = int(_get_arg(args, "num_layers", 2))
        self.dropout_rate = float(_get_arg(args, "dropout", 0.2))
        self.sim_threshold = float(_get_arg(args, "sim_threshold", 0.5))
        self.assoc_threshold = float(_get_arg(args, "assoc_threshold", 0.0))
        self.full_attention = bool(int(_get_arg(args, "full_attention", 1)))
        self.use_diffusion_bias = bool(int(_get_arg(args, "use_diffusion_bias", 1)))
        self.diffusion_bias_gain = float(_get_arg(args, "diffusion_bias_gain", 1.0))
        self.use_monotonic_mapping = bool(int(_get_arg(args, "use_monotonic_mapping", 1)))
        self.learn_scale_weight = bool(int(_get_arg(args, "learn_scale_weight", 1)))
        self.shared_diffusion = bool(int(_get_arg(args, "shared_diffusion", 0)))
        self.relation_fusion = str(_get_arg(args, "relation_fusion", "gate"))
        self.active_relations = self._parse_active_relations(_get_arg(args, "active_relations", "0,1,2"))
        self.use_pair_topology = bool(int(_get_arg(args, "use_pair_topology", 0)))
        self.neg_ratio = int(_get_arg(args, "neg_ratio", 1))
        self.negative_sampling_strategy = str(_get_arg(args, "negative_sampling_strategy", "random"))
        self.hard_negative_pool = int(_get_arg(args, "hard_negative_pool", 20))
        self.eval_chunk_size = int(_get_arg(args, "eval_chunk_size", 65536))

        self.topology_loss_weight = float(_get_arg(args, "topology_loss_weight", 0.1))
        self.alignment_loss_weight = float(_get_arg(args, "alignment_loss_weight", 1.0))
        self.centrality_loss_weight = float(_get_arg(args, "centrality_loss_weight", 1.0))

        seed = int(_get_arg(args, "seed", 42))
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.input_proj = nn.LazyLinear(self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_dropout = nn.Dropout(self.dropout_rate)

        self.bias_transform = MonotoneBiasTransform()
        self.scale_logits = nn.Parameter(torch.zeros(self.num_relations, self.num_scales))

        self.layers = nn.ModuleList(
            [
                FractionalRelationAttentionLayer(
                    hidden_dim=self.hidden_dim,
                    num_relations=self.num_relations,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate,
                    relation_fusion=self.relation_fusion,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )
        pair_input_size = self.output_dim * 4
        if self.use_pair_topology:
            pair_input_size += self.num_relations
        self.mlp_prediction = PairMLP(pair_input_size, self.dropout_rate)
        self.centrality_head = nn.Sequential(
            nn.Linear(self.output_dim, max(self.output_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(self.output_dim // 2, 16), 1),
        )

        self._graph_cache = None
        self._aux_loss = None
        self.last_relation_gates = None
        self.last_relation_bias = None
        self.last_embeddings = None
        self.last_graph_cache = None

    @staticmethod
    def _parse_active_relations(value):
        if value is None:
            return {0, 1, 2}
        names = {
            "drug": 0,
            "row": 0,
            "dd": 0,
            "microbe": 1,
            "col": 1,
            "mm": 1,
            "assoc": 2,
            "association": 2,
            "dm": 2,
        }
        active = set()
        for item in str(value).split(","):
            item = item.strip().lower()
            if not item:
                continue
            active.add(names[item] if item in names else int(item))
        return active or {0, 1, 2}

    def auxiliary_loss(self):
        if self._aux_loss is None:
            return next(self.parameters()).new_tensor(0.0)
        return self._aux_loss

    def _threshold_similarity(self, matrix):
        matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 0.0)
        return np.where(matrix > self.sim_threshold, matrix, 0.0).astype(np.float32)

    def _threshold_association(self, matrix):
        matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.where(matrix > self.assoc_threshold, matrix, 0.0).astype(np.float32)

    def _relation_adjacencies(self, rel_matrix, row_similarity, col_similarity):
        rel = _as_numpy(rel_matrix)
        row_sim = _as_numpy(row_similarity)
        col_sim = _as_numpy(col_similarity)
        left_nodes, right_nodes = rel.shape

        if row_sim.shape != (left_nodes, left_nodes) or col_sim.shape != (right_nodes, right_nodes):
            if row_sim.shape == (right_nodes, right_nodes) and col_sim.shape == (left_nodes, left_nodes):
                row_sim, col_sim = col_sim, row_sim
            else:
                raise ValueError(
                    "Similarity matrix shapes do not match the association matrix: "
                    f"rel={rel.shape}, row={row_sim.shape}, col={col_sim.shape}"
                )

        num_nodes = left_nodes + right_nodes
        row_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        col_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        assoc_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        row_adj[:left_nodes, :left_nodes] = self._threshold_similarity(row_sim)
        col_adj[left_nodes:, left_nodes:] = self._threshold_similarity(col_sim)
        assoc_block = self._threshold_association(rel)
        assoc_adj[:left_nodes, left_nodes:] = assoc_block
        assoc_adj[left_nodes:, :left_nodes] = assoc_block.T

        adjacencies = [row_adj, col_adj, assoc_adj]
        for rel_id in range(self.num_relations):
            if rel_id not in self.active_relations:
                adjacencies[rel_id] = np.zeros_like(adjacencies[rel_id])
        return adjacencies

    def _fractional_kernels(self, adjacency):
        adjacency = torch.from_numpy(adjacency).double()
        adjacency = 0.5 * (adjacency + adjacency.T)
        adjacency.fill_diagonal_(0.0)

        num_nodes = adjacency.size(0)
        degree = adjacency.sum(dim=1)
        active = degree > 0
        if not active.any():
            eye = torch.eye(num_nodes, dtype=torch.float32)
            return torch.stack([eye.clone() for _ in range(self.num_scales)], dim=0)

        active_index = torch.where(active)[0]
        sub_adj = adjacency[active][:, active]
        sub_degree = sub_adj.sum(dim=1)
        deg_inv_sqrt = torch.where(sub_degree > 0, sub_degree.rsqrt(), torch.zeros_like(sub_degree))
        norm_adj = deg_inv_sqrt[:, None] * sub_adj * deg_inv_sqrt[None, :]
        laplacian = torch.eye(sub_adj.size(0), dtype=torch.float64) - norm_adj

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        except torch._C._LinAlgError:
            jitter = 1e-7 * torch.eye(laplacian.size(0), dtype=laplacian.dtype)
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian + jitter)
        eigenvalues = eigenvalues.clamp_min(0.0)

        kernels = []
        for order in self.orders:
            fractional_values = eigenvalues.pow(float(order))
            for time in self.times:
                spectral_filter = torch.exp(-float(time) * fractional_values)
                sub_kernel = (eigenvectors * spectral_filter.unsqueeze(0)) @ eigenvectors.T
                sub_kernel = 0.5 * (sub_kernel + sub_kernel.T)
                kernel = torch.eye(num_nodes, dtype=torch.float64)
                kernel[active_index[:, None], active_index[None, :]] = sub_kernel
                kernels.append(kernel.float().contiguous())
        return torch.stack(kernels, dim=0)

    def _build_graph_cache(self, rel_matrix, row_similarity, col_similarity, device):
        adjacencies = self._relation_adjacencies(rel_matrix, row_similarity, col_similarity)
        kernels = torch.stack([self._fractional_kernels(adj) for adj in adjacencies], dim=0)

        num_nodes = kernels.size(-1)
        if self.full_attention:
            masks = torch.ones((self.num_relations, num_nodes, num_nodes), dtype=torch.bool)
        else:
            masks = torch.stack([torch.from_numpy(adj > 0) for adj in adjacencies], dim=0)
            eye = torch.eye(num_nodes, dtype=torch.bool).unsqueeze(0)
            masks = masks | eye

        mean_kernel = kernels.mean(dim=(0, 1))
        target_similarity = _normalize_01(mean_kernel)
        centrality = _standardize(mean_kernel.sum(dim=-1))

        self._graph_cache = {
            "num_nodes": num_nodes,
            "kernels": kernels.to(device=device, dtype=torch.float32),
            "masks": masks.to(device=device),
            "target_similarity": target_similarity.to(device=device, dtype=torch.float32),
            "centrality": centrality.to(device=device, dtype=torch.float32),
        }
        self.last_graph_cache = self._graph_cache
        return self._graph_cache

    def _get_graph_cache(self, rel_matrix, row_similarity, col_similarity, device):
        rel = _as_numpy(rel_matrix)
        num_nodes = rel.shape[0] + rel.shape[1]
        if self._graph_cache is not None and self._graph_cache["num_nodes"] == num_nodes:
            return self._graph_cache
        return self._build_graph_cache(rel, row_similarity, col_similarity, device)

    def _topology_bias(self, cache):
        if not self.use_diffusion_bias:
            kernels = cache["kernels"]
            return torch.zeros(
                (self.num_relations, kernels.size(-1), kernels.size(-1)),
                dtype=kernels.dtype,
                device=kernels.device,
            )
        kernels = cache["kernels"]
        if self.shared_diffusion:
            kernels = kernels.mean(dim=0, keepdim=True).repeat(self.num_relations, 1, 1, 1)
        transformed = self.bias_transform(kernels) if self.use_monotonic_mapping else kernels
        if self.learn_scale_weight:
            weights = torch.softmax(self.scale_logits, dim=-1).to(dtype=transformed.dtype, device=transformed.device)
        else:
            weights = torch.full(
                (self.num_relations, self.num_scales),
                1.0 / self.num_scales,
                dtype=transformed.dtype,
                device=transformed.device,
            )
        return self.diffusion_bias_gain * torch.sum(transformed * weights[:, :, None, None], dim=1)

    def get_scale_weights(self):
        if self.learn_scale_weight:
            return torch.softmax(self.scale_logits.detach().cpu(), dim=-1).numpy()
        return np.full((self.num_relations, self.num_scales), 1.0 / self.num_scales, dtype=np.float32)

    def _encode(self, x, cache):
        h = x.float()
        h = self.input_proj(h)
        h = self.input_norm(F.gelu(h))
        h = self.input_dropout(h)

        relation_bias = self._topology_bias(cache)
        self.last_relation_bias = relation_bias
        gates = []
        for layer in self.layers:
            h, gate = layer(h, relation_bias, cache["masks"])
            gates.append(gate)
        self.last_relation_gates = torch.stack(gates, dim=0).detach()
        embeddings = self.output_proj(h)
        self.last_embeddings = embeddings.detach()
        return embeddings

    def _compute_auxiliary_loss(self, embeddings, cache):
        if self.topology_loss_weight <= 0:
            return embeddings.new_tensor(0.0)

        ctx = torch.cuda.amp.autocast(enabled=False) if embeddings.is_cuda else nullcontext()
        with ctx:
            z = embeddings.float()
            normalized = F.normalize(z, p=2, dim=-1)
            embedding_similarity = (normalized @ normalized.T + 1.0) * 0.5
            target_similarity = cache["target_similarity"].to(device=z.device, dtype=z.dtype)
            alignment_loss = F.mse_loss(embedding_similarity, target_similarity)

            predicted_centrality = self.centrality_head(z).squeeze(-1)
            target_centrality = cache["centrality"].to(device=z.device, dtype=z.dtype)
            centrality_loss = F.mse_loss(predicted_centrality, target_centrality)

            return self.topology_loss_weight * (
                self.alignment_loss_weight * alignment_loss
                + self.centrality_loss_weight * centrality_loss
            )

    def _pair_features(self, left_embeddings, right_embeddings, topology_features):
        features = [
            left_embeddings,
            right_embeddings,
            left_embeddings * right_embeddings,
            torch.abs(left_embeddings - right_embeddings),
        ]
        if self.use_pair_topology:
            features.append(topology_features)
        return torch.cat(features, dim=-1)

    def _pair_topology_features(self, left_index, right_index, left_count):
        right_node_index = left_count + right_index
        return self.last_relation_bias[:, left_index, right_node_index].T

    def _sample_training_pairs(self, label_matrix, embeddings, forbidden_indices=None):
        if not isinstance(label_matrix, torch.Tensor):
            label_matrix = torch.from_numpy(np.asarray(label_matrix, dtype=np.float32)).to(embeddings.device)
        else:
            label_matrix = label_matrix.to(embeddings.device).float()

        left_count, right_count = label_matrix.shape
        left_embeddings = embeddings[:left_count]
        right_embeddings = embeddings[left_count:]

        positive_indices = torch.nonzero(label_matrix == 1, as_tuple=False)
        pos_left = positive_indices[:, 0]
        pos_right = positive_indices[:, 1]
        pos_topology = self._pair_topology_features(pos_left, pos_right, left_count)
        pos_features = self._pair_features(left_embeddings[pos_left], right_embeddings[pos_right], pos_topology)

        valid_negative = label_matrix == 0
        if forbidden_indices is not None:
            forbidden_indices = torch.as_tensor(forbidden_indices, dtype=torch.long, device=embeddings.device).view(-1, 2)
            if forbidden_indices.numel() > 0:
                rows = forbidden_indices[:, 0].clamp(0, left_count - 1)
                cols = forbidden_indices[:, 1].clamp(0, right_count - 1)
                valid_negative[rows, cols] = False

        neg_left_chunks = []
        neg_right_chunks = []
        col_degree = label_matrix.sum(dim=0)
        for row in pos_left.repeat_interleave(self.neg_ratio).tolist():
            candidates = torch.where(valid_negative[row])[0]
            if candidates.numel() == 0:
                raise ValueError(f"No valid negative samples remain for row {row}.")
            if self.negative_sampling_strategy == "degree_matched":
                weights = col_degree[candidates].float() + 1e-3
                selected = candidates[torch.multinomial(weights, 1)]
            elif self.negative_sampling_strategy == "hard":
                with torch.no_grad():
                    hard_scores = (right_embeddings[candidates] * left_embeddings[row].unsqueeze(0)).sum(dim=-1)
                    pool = min(self.hard_negative_pool, candidates.numel())
                    top_idx = torch.topk(hard_scores, pool).indices
                    selected = candidates[top_idx[torch.randint(0, pool, (1,), device=embeddings.device)]]
            else:
                selected = candidates[torch.randint(0, candidates.numel(), (1,), device=embeddings.device)]
            neg_left_chunks.append(torch.tensor([row], dtype=torch.long, device=embeddings.device))
            neg_right_chunks.append(selected)
        neg_left = torch.cat(neg_left_chunks, dim=0)
        neg_right = torch.cat(neg_right_chunks, dim=0)
        neg_topology = self._pair_topology_features(neg_left, neg_right, left_count)
        neg_features = self._pair_features(left_embeddings[neg_left], right_embeddings[neg_right], neg_topology)

        features = torch.cat([pos_features, neg_features], dim=0)
        labels = torch.cat(
            [
                torch.ones(pos_features.size(0), 1, device=embeddings.device),
                torch.zeros(neg_features.size(0), 1, device=embeddings.device),
            ],
            dim=0,
        )
        return features, labels

    def _predict_all_pairs(self, label_matrix, embeddings):
        if not isinstance(label_matrix, torch.Tensor):
            label_matrix = torch.from_numpy(np.asarray(label_matrix, dtype=np.float32)).to(embeddings.device)
        else:
            label_matrix = label_matrix.to(embeddings.device).float()

        left_count, right_count = label_matrix.shape
        left_embeddings = embeddings[:left_count]
        right_embeddings = embeddings[left_count:]
        outputs = []

        total_pairs = left_count * right_count
        for start in range(0, total_pairs, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, total_pairs)
            flat_index = torch.arange(start, end, device=embeddings.device)
            left_index = torch.div(flat_index, right_count, rounding_mode="floor")
            right_index = flat_index % right_count
            topology_features = self._pair_topology_features(left_index, right_index, left_count)
            pair_features = self._pair_features(left_embeddings[left_index], right_embeddings[right_index], topology_features)
            outputs.append(self.mlp_prediction(pair_features))

        logits = torch.cat(outputs, dim=0)
        labels = label_matrix.reshape(-1, 1)
        return logits, labels

    def forward(self, args, x, rel_matrix, SM, SD, index, L_normalized, train_model):
        del args, L_normalized
        label_matrix = rel_matrix if isinstance(rel_matrix, torch.Tensor) else np.asarray(rel_matrix, dtype=np.float32)
        x = x.to(self.device)
        cache = self._get_graph_cache(label_matrix, SM, SD, x.device)
        embeddings = self._encode(x, cache)

        if train_model:
            pair_features, labels = self._sample_training_pairs(label_matrix, embeddings, forbidden_indices=index)
            logits = self.mlp_prediction(pair_features.to(self.device))
            self._aux_loss = self._compute_auxiliary_loss(embeddings, cache)
            return logits, labels.to(self.device)

        logits, labels = self._predict_all_pairs(label_matrix, embeddings)
        self._aux_loss = embeddings.new_tensor(0.0)
        return logits, labels.to(self.device)
