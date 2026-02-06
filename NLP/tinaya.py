import torch,torch.nn.functional as f

def attention(q,k,v):
    d = q.size(-1)
    scores = q @ k.transpose(-2,-1) / d**0.5
    attn = f.softmax(scores,dim=-1)
    return attn @ v, attn

q = torch.randn(1,4,8)
out, attn = attention(q, q, q)
print(out.shape,attn.shape)

