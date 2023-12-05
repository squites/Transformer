import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# reproducibility
torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    tinyshakespeare = f.read()

# vocab has all unique characters from the dataset, so all possible characters to predict
vocab = sorted(list(set(tinyshakespeare)))
vocab_size = len(vocab)

# create mappings from characters to integers and vice-versa
def encode(chars):
    stoi = {ch:i for i,ch in enumerate(vocab)}
    return [stoi[c] for c in chars]

def decode(ints):
    itos = {i:ch for i,ch in enumerate(vocab)}
    return ''.join([itos[i] for i in ints])

#print(encode('transformers!'))
#print(decode(encode('transformers!')))

# Train and test splits
data = torch.tensor(encode(tinyshakespeare), dtype=torch.long) # encode the whole dataset
n = int(0.9*len(data))
train_data = data[:n] # 90%
val_data = data[n:]   # 10%

# generate a random batch of data of inputs x and targets y
def get_batch(split):
    #data = train_data if split == 'train' else val_data
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MaskedMultiheadAttention(nn.Module):
    """
        Computes MultiHead self-attention in parallel, treating heads as a dimension
    """
    def __init__(self, n_heads): # 32, 8 # There was head_size as a parameter too
        super().__init__()
        assert n_embd % n_heads == 0 # 384 % 6 == 0 -> True
        self.Wk = nn.Linear(n_embd, n_embd, bias=False) # (384, 384)
        self.Wq = nn.Linear(n_embd, n_embd, bias=False) # (384, 384)
        self.Wv = nn.Linear(n_embd, n_embd, bias=False) # (384, 384)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
        # residual connections and dropout regularization for efficiency
        self.skip_connection = nn.Linear(n_embd, n_embd) # (384, 384)
        self.affinities_drop = nn.Dropout(dropout)
        self.residual_drop = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # B:64, T:256, C:384
        # These are the actual vectors
        k = self.Wk(x) # (64,256,384) @ (384, 384) -> (64,256,384)
        q = self.Wq(x) # (64,256,384) @ (384, 384) -> (64,256,384)
        v = self.Wv(x) # (64,256,384) @ (384, 384) -> (64,256,384)

        # splitting k,q,v into their heads changing their shapes
        k = k.view(B, T, n_heads, C//n_heads).transpose(-3, -2) # (64,6,256,64) # 384/6=64
        q = q.view(B, T, n_heads, C//n_heads).transpose(-3, -2)
        v = v.view(B, T, n_heads, C//n_heads).transpose(-3, -2)

        # Scaled Dot-Product attention -----
        # (64,6,256,64) @ (64,6,64,256) -> (64,6,256,256)
        affinities = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #(k.shape[-1]**-0.5)
        affinities = affinities.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        affinities = self.affinities_drop(affinities)

        # weight aggregation with values
        aggregation = affinities @ v # (64,6,256,256) @ (64,6,256,64) -> (64,6,256,64)

        # reshape back
        aggregation = aggregation.transpose(-3, -2).contiguous().view(B,T,C) # (64,256,6,64)

        # residual connection and dropout
        out = self.residual_drop(self.skip_connection(aggregation))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        #self.linear1 = nn.Linear(n_embd, 4*n_embd)
        #self.relu = nn.ReLU()
        #self.linear2 = nn.Linear(4*n_embd, n_embd)
        #self.drop = nn.Dropout(dropout)

    def forward(self, x):
        #out = self.linear1(x)
        #out = self.relu(x)
        #out = self.linear2(x)
        #out = self.drop(out)
        return self.net(x)#out

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_heads):
        # n_embd: embedding dimension, n_heads: the number of heads we'd like
        super().__init__()
        #head_size = n_embd // n_heads
        self.attn = MaskedMultiheadAttention(n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Training the model
model = GPTLanguageModel()
m = model.to(device)

# Number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Training in 48 min. Let's try to improve that!
# It's underfitting. Maybe lower the layer numbers, increase the dataset
# step 4999: train loss 0.8590, val loss 1.5759

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))