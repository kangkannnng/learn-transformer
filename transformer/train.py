import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from transformer.transformer_stucture import Transformer

# 定义模型参数
d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)
model = Transformer(src_vocab, trg_vocab, d_model, N, heads)

for p in model.parameters():
    if p.dim() > 1:
        # 初始化权重
        nn.init.xavier_uniform_(p)

# 定义优化器
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 模型训练
def train_model(epochs, print_every=100):
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            src = batch.English.transpose(0, 1)
            trg = batch.French.transpose(0, 1)
            # 去掉句子最后一个字符，因为最后一个字符不需要预测
            trg_input = trg[:, :-1]
            # 预测目标
            targets = trg[:, 1:].contiguous().view(-1)
            # mask掉padding部分
            src_mask, trg_mask = create_mask(src, trg_input)
            # 计算模型输出
            preds = model(src, trg_input, src_mask, trg_mask)
            optim.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), 
                                   results, 
                                   ignore_index=target_pad)
            loss.backward()
            optim.step()
            total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % 
                      ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg,
                        time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()

# 模型测试
def translate(model, src, max_len=80, custom_sentence=False):
    model.eval()
    if custom_sentence==True:
        src = tokenize_en(src)
        sentence=torch.tensor(torch.LongTensor([EN_TEXT.vocab.stoi[i] for i in src]))
    src_mask = (src != EN_TEXT.vocab.stoi['<pad>']).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)
    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']])

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones((1, i, i)).astype('uint8'))
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0)
        out = model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask)
        out = model.out(out)
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)
        outputs[i] = ix[0][0]
        if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
            break
    return ' '.join([FR_TEXT.vocab.itos[ix] for ix in outputs[:i]])