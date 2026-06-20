import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import time

torch.manual_seed(66)

raw_data = [
    ("I love machine learnning","我 喜欢 机器 学习"),
    ("Deep learnninng is powerful","深度 学习 很 强大"),
    ("Transformer changed everything","Transformer 改变 了 一切"),
    ("你 不会 游泳 ， 是 我 救的 你 。","我 会 。"),
    ("你 说 你 不会 。","嗯嗯 ， 我 不会 。")
]

def build_vocab(sentences,min_freq=1):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    vocab = {"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word]=len(vocab)
    return vocab

def encode(sentence, vocab,max_len=10):
    tokens = sentence.split()
    ids = [vocab.get(tok,vocab["<unk>"]) for tok in tokens]
    ids = [vocab["<bos>"]] + ids + [vocab["<eos>"]]
    if len(ids)<max_len:
        ids += [vocab["<pad>"]]*(max_len-len(ids))
    else:
        ids = ids[:max_len]
    return ids

# 点积计算
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, q, k, v):
        ##
        # q: [batch_size, n_heads, len_q, d_k]
        # k: [batch_size, n_heads, len_k, d_k]
        # v: [batch_size, n_heads, len_v, d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        ##
        # 计算每个Q与K的分数，计算出来的大小是 [batch_size, n_heads, len_q, len_q]
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 把被mask的地方置为无限小，softmax之后基本就是0，也就对q不起作用
        # scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # 注意力后的大小 [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, v)
        return context, attn


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        ##
        # q: [batch_size, seq_len, d_model]
        # k: [batch_size, seq_len, d_model]
        # v: [batch_size, seq_len, d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        ##
        # 记录原始值, 后续计算残差
        residual, batch_size = q, q.size(0)
        # 先映射 q、k、v, 然后后分头；
        # q: [batch_size, n_heads, len_q, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k: [batch_size, n_heads, len_k, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v: [batch_size, n_heads, len_v(=len_k), d_v]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # 点积注意力分数计算，  [batch_size, n_heads, len_q, d_v]
        context, attn = ScaledDotProductAttention(self.d_k)(q, k, v)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 还原为原始大小
        output = self.fc(context)
        # LN + 残差计算
        return self.layernorm(output + residual), attn


# 门控网络
class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        # 用于进行惩罚计算的噪声
        self.noise_linear = nn.Linear(d_model, num_experts)
    def forward(self, x):
        logits = self.gate(x)
        # 训练时添加噪声
        if self.training:
            noise = torch.randn_like(logits).to(x.device)
            noise = self.noise_linear(x) * noise
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
        gates_prob = F.softmax(noisy_logits, dim=-1)
        # Top-k 选择
        top_k_probs, top_k_indices = torch.topk(gates_prob, self.top_k, dim=-1)
        # 归一化，确保被选中的专家的权重之和为1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        # 专家惩罚损失
        expert_penalty_loss = self.compute_expert_penalty_loss(gates_prob, top_k_indices)

        return top_k_probs, top_k_indices, expert_penalty_loss

    def compute_expert_penalty_loss(self, gates_prob, top_k_indices):
        """ 专家惩罚损失：num_experts * sum ( 每个专家的平均概率 * 每个专家选中的概率 )"""
        # 计算每个专家的平均概率
        router_prob_per_expert = gates_prob.mean(dim=(0, 1))
        # 计算每个专家理想被分配到的概率
        expert_mask = torch.zeros_like(gates_prob)
        expert_mask.scatter_(2, top_k_indices, 1)
        tokens_per_expert = expert_mask.float().mean(dim=(0, 1))
        # 惩罚损失
        return self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)
    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


# 专家网络
# Inspired by my COLM manuscript
class Expert(nn.Module):
    def __init__(self, d_model):
        super(Expert, self).__init__()
        self.gate = nn.Sequential(nn.Linear(d_model,d_model),nn.SiLU())
        self.Q = ComplexLinear(144,144)
        self.K = ComplexLinear(144,144)
        self.V = ComplexLinear(144,144)
        self.w_o = nn.Linear(d_model,d_model)

    def forward(self, x):
        gate = self.gate(x)
        x_spatial = x.reshape(1,30,16,16)
        x_fft = torch.fft.rfft2(x_spatial,dim=(1,2),norm='ortho')#(复数)归一化因子为1/sqrt(n)（使实数FFT正交化）
        x_fft = x_fft.reshape(1,30,144)
        F_q = self.Q(x_fft)
        F_k = self.K(x_fft)
        F_v = self.V(x_fft)
        attn_fft = torch.conj(F_q)*F_k
        attn_fft = attn_fft.reshape(1,30,9,16)
        attn_spatial = torch.fft.irfft2(attn_fft,dim=(1,2),norm='ortho')
        attn_weight = attn_spatial.reshape(1,768,10).softmax(dim=1).reshape(1,30,16,16)#注意力权重归一化
        attn_fft = torch.fft.rfft2(attn_weight,dim=(1,2))
        attn_fft = attn_fft.reshape(1,30,144)
        v_weighted_fft=attn_fft*F_v
        v_weighted_fft=v_weighted_fft.reshape(1,30,9,16)
        v_weighted_spatial = torch.fft.irfft2(v_weighted_fft,dim=(1,2),norm='ortho')
        v_weighted_spatial = v_weighted_spatial.reshape(1,10,768)
        out_seq = v_weighted_spatial*gate #门控加权
        res = self.w_o(out_seq)
        return res


# MOE层
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super(MoELayer, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        # 门控路由，决定哪些专家被激活
        self.router = Router(d_model, num_experts, top_k)
        # 创建多个专家
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        # Layer Norm
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        residual = x
        batch_size, seq_len, d_model = x.shape
        # 获取路由决策
        # gates: [batch_size, seq_len, top_k]
        # selected_experts: [batch_size, seq_len, top_k]
        gates, selected_experts, expert_penalty_loss = self.router(x)
        # 初始化输出
        output = torch.zeros_like(x)
        # 对每个token应用选中的专家
        for i in range(self.top_k):
            # 获取当前专家索引
            expert_idx = selected_experts[:, :, i]  # [batch_size, seq_len]
            # print("per expert shape:",expert_idx.shape)
            # 获取当前权重
            expert_gate = gates[:, :, i]  # [batch_size, seq_len]
            # 对每个专家进行计算
            for expert_id in range(self.num_experts):
                # 找出选择了当前专家的token位置
                mask = (expert_idx == expert_id).unsqueeze(-1)  # [batch_size, seq_len, 1]
                if mask.any():
                    # 获取分配给当前专家的tokens
                    expert_input = x * mask  # [batch_size, seq_len, d_model]
                    expert_output = self.experts[expert_id](expert_input)  # [batch_size, seq_len, d_model]
                    # print("expert output shape:",expert_output.shape)
                    # 加权输出
                    weighted_output = expert_output * expert_gate.unsqueeze(-1) * mask
                    output += weighted_output
        # 残差连接和Layer Norm
        output = self.layernorm(output + residual)
        return output, expert_penalty_loss

# 解码层
class MoEDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, num_experts=8, top_k=2):
        super(MoEDecoderLayer, self).__init__()
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        # MoE
        self.pos_ffn = MoELayer(d_model, num_experts, top_k)

    def forward(self, inputs):
        # 多头注意力
        outputs, self_attn = self.attention(inputs, inputs, inputs)
        # MoE
        outputs, expert_penalty_loss = self.pos_ffn(outputs)
        return outputs, self_attn, expert_penalty_loss


# 位置编码，这里使用GPT2的做法
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_pos, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(max_pos, d_model)
    def forward(self, inputs):
        seq_len = inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(inputs)
        return self.pos_embedding(pos)

# 解码器
class MoEDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, vocab_size, max_pos, n_layers,
                 device, num_experts=8, top_k=2):
        super(MoEDecoder, self).__init__()
        self.device = device
        # 将Token转为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_pos, device)
        # 创建MOE层
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(MoEDecoderLayer(d_model, n_heads, d_k, d_v,num_experts, top_k))

    def forward(self, inputs):
        # 嵌入和位置编码
        outputs = self.embedding(inputs) + self.pos_encoding(inputs)
        # 计算每一层的结果
        self_attns = []
        total_expert_penalty_loss = 0.0
        for layer in self.layers:
            layer_output = layer(outputs)
            outputs, self_attn, expert_penalty_loss = layer_output
            total_expert_penalty_loss += expert_penalty_loss
            self_attns.append(self_attn)

        return outputs, self_attns, total_expert_penalty_loss

# GPT MOE模型
class GPTMoEModel(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, vocab_size, max_pos, n_layers,
                 device, num_experts=8, top_k=2, expert_penalty_factor=0.01):
        super(GPTMoEModel, self).__init__()
        self.expert_penalty_factor = expert_penalty_factor
        # 解码器
        self.decoder = MoEDecoder(d_model, n_heads, d_k, d_v, vocab_size, max_pos, n_layers,device, num_experts, top_k)
        # 映射为词表大小
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, inputs):
        # 前向传播
        outputs, self_attns, expert_penalty_loss = self.decoder(inputs)
        # 投影到词表
        logits = self.projection(outputs)
        logits = logits.view(-1, logits.size(-1))
        expert_penalty = expert_penalty_loss*self.expert_penalty_factor

        return logits, self_attns,expert_penalty


if __name__ == '__main__':
    device = "cpu"
    sentence_lis=[]
    for s in raw_data:
        sentence_lis.append(s[0])
        sentence_lis.append(s[1])
    src_vocab = build_vocab(sentence_lis)
    # print(len(src_vocab))
    # print(src_vocab)
    input_seqs=[encode(s[0],src_vocab,max_len=10) for s in raw_data]
    target_seqs=[encode(s[1],src_vocab,max_len=10) for s in raw_data]

    input_tensor = torch.tensor([input_seqs[0]],dtype=torch.long)
    label_tensor = torch.tensor([target_seqs[0]],dtype=torch.long) 
    pre_mask = torch.zeros_like(input_tensor)
    # print(input_tensor)
    for i in range(input_tensor.shape[1]):
        if input_tensor[0][i]!=0:
            pre_mask[0][i]=1

    model_param = {
        "d_model": 768,  # 嵌入层大小
        "d_k": 64,  # K 的大小
        "d_v": 64,  # V 的大小
        "n_layers": 6,  # 解码层的数量
        "n_heads": 8,  # 多头注意力的头数
        "max_pos": 10000,  # 位置编码的长度
        "device": device,  # 设备
        "vocab_size": 35,  # 词表大小
        "num_experts": 8,  # 8个专家
        "top_k": 2,  # 每个token选择2个专家
        "expert_penalty_factor": 0.01  # 专家惩罚因子
    }
    model = GPTMoEModel(**model_param)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss()
    # outputs, dec_self_attns, loss = model(input_tensor, pre_mask)
    # print(outputs.shape)
    # print(len(dec_self_attns))
    # print(loss)
    
    for epoch in range(200):
        start=time.time()
        epoch_avg_loss=0.0
        epoch_avg_ep = 0
        batch_count=0
        for i in range(len(input_seqs)):
            input_tensor = torch.tensor([input_seqs[i]])
            label_tensor = torch.tensor([target_seqs[i]]) 
            optimizer.zero_grad()
            outputs, dec_self_attns, ep_loss = model(input_tensor)
            # logits = logits.reshape(-1,logits.size(-1))
            logits_loss = loss_fn(outputs,label_tensor.squeeze(0))
            loss =logits_loss+ep_loss
            # loss =logits_loss
            # loss = loss_fn(logits.squeeze(0)[0],label_tensor[0][0])
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_avg_loss+=logits_loss
            epoch_avg_ep+=ep_loss
            batch_count+=1
            # print("logits loss:",logits_loss.detach().cpu().numpy()," expert penalty loss:",ep_loss.detach().cpu().numpy())
        end=time.time()
        epoch_avg_loss/=batch_count
        epoch_avg_ep/=batch_count
        print("epoch:",epoch," epoch_loss:",epoch_avg_loss.detach().cpu().numpy()," epoch_avg_ep:",epoch_avg_ep.detach().cpu().numpy()," time:",end-start,"s")