import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

# 测试的时候固定随机数种子
torch.manual_seed(0)

# 定义位置编码器
class PositonalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        # 定义一个矩阵positionalencoder，矩阵的大小是max_seq_len * d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # 加入位置编码
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        # 增加一个维度，因为x是一个三维的张量[batch_size, seq_len, d_model]
        pe = pe.unsqueeze(0)
        # 注册pe为buffer，这样在训练的时候不会被更新
        self.register_buffer('pe', pe)

    # 前向传播函数
    def forward(self, x):
        # 这里的乘法是为了让单词嵌入更大一些，这样在加上位置编码之后，不会因为位置编码太小而影响到单词嵌入
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        # 控制长度匹配
        x = x + self.pe[:,:seq_len].requires_grad_(False)
        return x


# 定义多头注意力模型
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        # d_k表示每个头的k的维度
        self.d_k = d_model // heads
        self.h = heads
        # 定义三个权重矩阵，分别是q、k、v，Linear是一个全连接层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        # dropout层是为了防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 输出层也是一个全连接层
        self.out = nn.Linear(d_model, d_model)

    # 定义注意力函数
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # scores是q和k的乘积，除以根号d_k
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 如果mask不为空，则将mask中为0的位置的score设置为一个很小的值，这样softmax之后就为0
        if mask is not None:
            # 这里是为了将mask扩展到和scores一样的维度，然后将mask中的值为0的位置的score设置为一个很小的值
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)

        # 对scores进行softmax操作，然后使用dropout
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores, v)
        return output
    
    # 前向传播函数
    def forward(self, q, k, v, mask=None):
        # q的维度是[batch_size, seq_len, d_model]
        bs = q.size(0)

        # view函数用来改变张量的形状，self.h表示多头的个数
        # 此时四个维度分别是batch_size, seq_len, self.h, self.d_k
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose函数用来交换张量的维度，1和2进行交换是为了将多头放到前面
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # 最后再拼接成最初的维度[batch_size, seq_len, d_model]
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        # 输出层作用是为了将多头注意力的结果映射到原始的维度
        output = self.out(concat)
        return output


# 定义FNN层
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # 线性层1，将d_model映射到d_ff，这个维数要比d_model大
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    # 前向传播函数
    def forward(self, x):
        # 这里是先通过一个线性层，然后使用relu激活函数，最后再通过另一个线性层
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
    

# 定义残差连接和层归一化，目的是为了防止梯度消失
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # 定义两个参数，分别是α和β
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    # 前向传播函数
    def forward(self, x):
        # 计算x的均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 归一化
        norm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return norm
    
# 定义编码器结构
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        # 定义两个norm是因为在多头注意力和FNN之后都有一个残差连接和层归一化
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    # 前向传播函数
    def forward(self, x, mask):
        # 这里四个参数分别是q, k, v, mask
        attn_output = self.attn(x, x, x, mask)
        attn_output = self.dropout_1(attn_output)
        # 残差连接和层归一化
        x = x + attn_output
        x = self.norm_1(x)
        # FNN
        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output
        x = self.norm_2(x)
        return x


# 定义解码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositonalEncoder(d_model, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(N)])
        self.norm = Norm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return x

# 定义解码器结构
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        # 定义三个norm是因为在三个地方都有一个残差连接和层归一化
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    # 前向传播函数
    def forward(self, x, e_outputs, src_mask, trg_mask):
        # 第一个多头注意力，这是自注意力
        attn_output_1 = self.attn_1(x, x, x, trg_mask)
        attn_output_1 = self.dropout_1(attn_output_1)
        x = x + attn_output_1
        x = self.norm_1(x)
        # 第二个多头注意力，这是交叉attention
        attn_output_2 = self.attn_2(x, e_outputs, e_outputs, src_mask)
        attn_output_2 = self.dropout_2(attn_output_2)
        x = x + attn_output_2
        x = self.norm_2(x)
        # FNN
        ff_output = self.ff(x)
        ff_output = self.dropout_3(ff_output)
        x = x + ff_output
        x = self.norm_3(x)
        return x


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositonalEncoder(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(N)])
        self.norm = Norm(d_model)
    
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return x
    
# 定义Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
    

# 测试
# 使用data文件夹下的english.txt和french.txt
EN_TEXT = Field(tokenize = 'spacy', tokenizer_language='en', init_token = '<sos>', eos_token = '<eos>', lower = True)
FR_TEXT = Field(tokenize = 'spacy', tokenizer_language='fr', init_token = '<sos>', eos_token = '<eos>', lower = True)

# TabularDataset加载数据
train_data, valid_data, test_data = TabularDataset.splits(
    path = 'data',
    train = 'english.txt',
    validation = 'french.txt',
    test = 'french.txt',
    format = 'tsv',
    fields = [('English', EN_TEXT), ('French', FR_TEXT)]
)

# 构建词汇表
EN_TEXT.build_vocab(train_data, min_freq = 2)
FR_TEXT.build_vocab(train_data, min_freq = 2)

# 创建迭代器
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = torch.device('cuda')
)


# 这里还是有问题！！！

d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)
model = Transformer(src_vocab, trg_vocab, d_model, N, heads, dropout=0.1).to(torch.device('cuda'))
for p in model.parameters():
    if p.dim() > 1:
        # 这是为了防止梯度爆炸
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = torch.tensor(np_mask) == 0
    return np_mask

# 定义一个mask矩阵，这个矩阵是一个上三角矩阵，对角线以下的值都是0，对角线以上的值都是1
def create_masks(src, trg, opt):
    
    src_mask = (src != opt.src_pad).unsqueeze(-2).to(opt.device)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2).to(opt.device)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt).to(opt.device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask

def train_model(epochs, print_every=100):
    model.train()
    start = time.time()
    temp = start
    total_loss = 0
    for epoch in range(epochs):
        for i, batch in enumerate(train_iterator):
            src = batch.English.transpose(0, 1).to(torch.device('cuda'))
            trg = batch.French.transpose(0, 1).to(torch.device('cuda'))
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask, trg_mask = create_masks(src, trg_input)
            