import torch
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
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

class RMSNorm(nn.Module):
    def __init__(self, dim:int,eps:float = 1e-6):
        super().__init__()
        # 可训练参数gamma
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        # 计算平方根
        x=x.float()
        rms = torch.sqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        # 归一化并缩放
        return self.gamma*x/rms
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(256,300)
        self.fc2=nn.Linear(256,300)
        self.fc3=nn.Linear(300,256)
    
    def forward(self,x):
        x_fc1=self.fc1(x)
        x_fc2=self.fc2(x)
        x_dot = F.silu(x_fc1)*x_fc2
        out = self.fc3(x_dot)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in=256,num_heads=8):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=d_in//num_heads
        self.d_out = num_heads*self.head_dim

        self.W_query = nn.Linear(d_in,self.d_out)
        self.W_key = nn.Linear(d_in,self.d_out)
        self.W_value=nn.Linear(d_in,self.d_out)

        self.out_proj = nn.Linear(self.d_out,d_in)
        self.rope_theta = 10000.0
        inv_freq = 1.0/(self.rope_theta**(torch.arange(0,self.head_dim,2).float()/self.head_dim))
        pos = torch.arange(10).float()
        freqs = torch.einsum('i,j->ij',pos,inv_freq)
        self.register_buffer("freqs_cis",torch.polar(torch.ones_like(freqs),freqs).unsqueeze(0).unsqueeze(2))
    
    def forward(self,x):#,mask=None
        b,num_tokens,_=x.shape
        quieries = self.W_query(x)
        keys=self.W_key(x)
        values=self.W_value(x)
        quieries=quieries.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        key_new = keys.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        values_new = values.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)

        # RoPE旋转位置编码
        def apply_rope(tensor,n_neads):
            t = tensor.reshape(b,num_tokens,n_neads,self.head_dim//2,2)
            t = torch.view_as_complex(t)*self.freqs_cis[:,:num_tokens]
            res = torch.view_as_real(t).flatten(3)
            return res
        # RoPE应用位置（遵循chain rule）
        quieries = apply_rope(quieries,self.num_heads)
        key_new=apply_rope(key_new,self.num_heads)

        attn_scores = quieries@key_new.transpose(2,3)
        # attn_scores = attn_scores.masked_fill(mask[:,:,:,:].to("cuda:0")==0,1e-9)
        attn_weights = torch.softmax(attn_scores/self.head_dim**0.5,dim=-1)
        context = attn_weights@(values_new.transpose(1,2))
        context = context.reshape(b,num_tokens,self.d_out)
        final_out = self.out_proj(context)

        return final_out
    
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=MultiHeadAttention()
        self.ff=FeedForward()
        self.norm1=RMSNorm(256)
    
    def forward(self,x):#,mask
        x=self.attn(x)#,mask
        x = self.norm1(x)
        res = self.ff(x)
        res_norm=self.norm1(res)
        return res_norm

        

class LLM_base(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_emb=nn.Embedding(35,256)
        self.trf_blocks = nn.ModuleList([TransformerBlock() for _ in range(6)])
        self.final_norm = RMSNorm(256)
        self.out=nn.Linear(256,35)
    
    def forward(self,x,cache=None):
        word_embed = self.word_emb(x)
        x=word_embed
        num_tokens = x.shape[1]
        # mask=torch.triu(torch.ones(num_tokens,64),diagonal=1)
        # mask = mask.reshape(1,10,8,8)
        for i,block in enumerate(self.trf_blocks):
            # x = block(x,mask)
            x = block(x)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits


if __name__=='__main__':
    sentence_lis=[]
    for s in raw_data:
        sentence_lis.append(s[0])
        sentence_lis.append(s[1])
    src_vocab = build_vocab(sentence_lis)
    print(src_vocab)
    input_seqs=[encode(s[0],src_vocab,max_len=10) for s in raw_data]
    target_seqs=[encode(s[1],src_vocab,max_len=10) for s in raw_data]
    # print(input_seqs)
    # print(target_seqs)
    vocab_size=len(src_vocab)
    # print(vocab_size)
    d_model = 256
    
    learning_rate = 0.0005
    batch_size=1
    epochs=300
    model = LLM_base().to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        start = time.time()
        epoch_loss=0.0
        batch_count=0
        for i in range(len(input_seqs)):
            input_tensor = torch.tensor([input_seqs[i]],dtype=torch.long)
            label_tensor = torch.tensor([target_seqs[i]],dtype=torch.long) 
            input_tensor=input_tensor.to("cuda:0")
            label_tensor = label_tensor.to("cuda:0")
            optimizer.zero_grad()
            logits = model(input_tensor)
            loss = loss_fn(logits.squeeze(0),label_tensor.squeeze(0))
            loss.backward()
            optimizer.step()
            epoch_loss+=loss
            batch_count+=1
            
        epoch_loss/=batch_count
        end=time.time()
        print("epoch:",epoch,"loss:",epoch_loss.detach().cpu().numpy(),"time:",end-start,"s")
    #     # break
    # print(len(src_vocab))
    torch.save(model,"LLM_base_epoch300.pth")
    # res_dict=torch.load('Libra_epoch300_p_gen.pth')
    # print("=====")
    # input_tensor = torch.tensor([input_seqs[2]],dtype=torch.long)
    # output_tensor= torch.tensor([target_seqs[2]],dtype=torch.long) 
    # print("input question:",input_seqs[2])
    # print("output tensor:",output_tensor)
    # print(raw_data[2][0])
    # logits = res_dict(input_tensor)

    # result=torch.argmax(logits.squeeze(0),dim=1)
    # print(result)
    # print(logits.shape)

    # result = torch.argmax(logits.squeeze(0), dim=-1)
    # print(result)
    # id2word={}
    # for item in src_vocab:
    #     id2word[src_vocab[item]]=item
    # print(id2word)
    # question=[]
    # print(input_tensor)
    # for i in range(len(input_tensor[0])):
    #     question.append(id2word[input_tensor[0][i].detach().numpy().tolist()])
    # print("ques:")
    # print(" ".join(question))
    # res=[]
    # for i in range(len(result)):
    #     res.append(id2word[result[i].detach().numpy().tolist()])
    # print("res:")
    # print(" ".join(res))

