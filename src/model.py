import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torcheval.metrics.functional import r2_score

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy: (N, heads, query_len, key_len)

        if mask is not None:
            # 当mask中的值为0时，对应位置的energy值会被替换为float("-1e20")
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then flatten out the last two dimensions.

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_feature_dim,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,     # 时间片长度
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_feature_linear = nn.Linear(src_feature_dim, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, T, C = x.shape
        positions = torch.arange(0, T).expand(N, T).to(self.device)
        a = self.position_embedding(positions)
        b = self.src_feature_linear(x)
        out = self.dropout(a + b)
        # out = self.dropout(
        #     (self.src_feature_linear(x) + self.position_embedding(positions))
        # )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_feature_dim,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        # self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_feature_linear = nn.Linear(trg_feature_dim, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, T, C = x.shape
        positions = torch.arange(0, T).expand(N, T).to(self.device)
        x = self.dropout((self.trg_feature_linear(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_feature_dim,
        trg_feature_dim,
        src_pad_idx,
        trg_pad_idx,
        max_length,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_feature_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length
        )

        self.decoder = Decoder(
            trg_feature_dim, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.max_length = max_length

    def make_src_mask(self, src):
        # src shape: (N, src_len)
        # src_mask shape: (N, 1, 1, src_len)
        # 确保了模型在计算注意力得分和后续操作时能够忽略填充位置的数据，从而保持对实际有效序列内容的正确处理和学习
        src_mask = (src[:,:,0] != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, T, V = trg.shape
        # 创建一个目标序列的掩码，以确保在解码时，模型只能看到当前及之前的位置，不能看到未来的位置。使用下三角矩阵来实现
        trg_mask = torch.tril(torch.ones((T, T))).expand(
            N, 1, T, T
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src = self.pad_sequences(src, self.src_pad_idx, self.max_length)
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, None)
        loss = None
        if trg is not None:
            loss = F.mse_loss(out.view(-1, 2), trg.view(-1, 2))
            r2_s = r2_score(out.view(-1, 2), trg.view(-1, 2))
        return out, loss, r2_s

    def get_optimizer_and_scheduler(self, config):
        # 基础的Adam优化器
        optimizer = Adam(self.parameters(), lr=config.learningRate, betas=config.betas, eps=config.eps)

        # 学习率随训练步骤动态调整的函数
        def lr_lambda(current_step: int):
            num_warmup_steps = config.warmup_steps
            num_training_steps = config.total_steps
            warmup = min(current_step / num_warmup_steps, 1.0)
            decay = (num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps)
            return warmup * decay

        # 学习率调度器
        scheduler = LambdaLR(optimizer, lr_lambda)

        return optimizer, scheduler

    def pad_sequences(self, batch, src_pad_idx, max_length):
        """
        对给定的批次数据进行填充，以确保所有序列的长度一致。

        参数:
        - batch: 输入的批次数据，假设形状为[N, T, C]，其中T可能小于128。
        - pad_value: 用于填充的值，默认为-1。

        返回:
        - padded_batch: 填充后的批次数据，形状为[N, 128, C]。
        """
        N, T, C = batch.shape
        if T == max_length:
            return batch  # 如果序列长度已经是128，则不需要填充

        # 计算需要填充的长度
        pad_length = max_length - T

        # 创建填充用的数组
        pad_tensor = torch.full((N, pad_length, C), src_pad_idx, dtype=batch.dtype, device=batch.device)

        # 将原始数据和填充数据拼接在一起
        padded_batch = torch.cat([batch, pad_tensor], dim=1)

        return padded_batch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 16).to(device)
    out = model(x, trg[:, :-1]) # 切片操作，:-1表示选取从第一个元素到倒数第二个元素，不包括序列中的最后一个元素
    print(out.shape)