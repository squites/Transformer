import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64#32 #4 #64
block_size = 256#10 #8 #256
n_embd = 384#512 #32 #384
n_heads = 6#8 #4 #6
#head_size = 16
n_layers = 6
dropout = 0.2
learning_rate = 3e-4
max_iterations = 5000
eval_interval = 500
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# reproducibility
torch.manual_seed(1337)

# download the data: "!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
with open('./input.txt', 'r', encoding='utf-8') as f:
    tinyshakespeare = f.read()

# vocab has all unique characters from the dataset, so all possible characters to predict
vocab = sorted(list(set(tinyshakespeare)))
vocab_size = len(vocab)

# maps from character to integer and vice-versa
def encode(chars):
    stoi = {ch:i for i,ch in enumerate(vocab)}
    return [stoi[c] for c in chars]

def decode(ints):
    itos = {i:ch for i,ch in enumerate(vocab)}
    return ''.join([itos[i] for i in ints])

print(encode('transformers!'))
print(decode(encode('transformers!')))

# train/test splits
data = torch.tensor(encode(tinyshakespeare), dtype=torch.long) # encode the whole tinyshakespeare dataset
n = int(0.9 * len(data))
train_data = data[:n] # 90%
val_data   = data[n:] # 10%

# generate a random batch of data with inputs x and targets y
def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ixs = torch.randint(len(data) - block_size, (batch_size,)) # generate 32 batches of 10 block_size (Ex.)
    x = torch.stack([data[i:i+block_size] for i in ixs])
    y = torch.stack([data[i+1:i+block_size+1] for i in ixs])
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
"""
# one head of self-attention
class SelfAttentionHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False) # (32,head_size)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape     # (batch_size, block_size, n_embd) -> (4,8,32) 
        k = self.key(x)     # (batch, block, n_embd) @ (n_embd, head_size) -> (batch, block, head_size)
        q = self.query(x)
        v = self.value(x)

        # compute the "affinities"
        affn = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, head_size) @ (B, head_size, T) -> (B,T,T)
        affn = affn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        affn = F.softmax(affn, dim=-1)
        #affn = self.dropout(affn)
        
        # weight aggregation
        out = affn @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)

        return out
"""
"""
# multiple self-attention heads
class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size): # (head_size)
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_heads)]) # creates n_heads SelfAttentionHeads and groups into a list
        #self.proj = nn.Linear(head_size * n_heads, n_embd) # (16*4, 32) -> (64, 32)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) 
        #out = self.proj(out)
        #out = self.dropout(out)
        return out
"""

# trying to make multi-head class without having to do a self-attention class, so treating heads as a dim
class MultiHeadAttention_attempt(nn.Module):

    def __init__(self, head_size, n_heads): # 32, 8
        super().__init__()
        assert n_embd % n_heads == 0 # 512%8 == 0 -> True
        #self.heads = n_heads # 8
        #(DEBUG)print(f"n_embd, head_size: {n_embd}, {head_size}")
        # note: changed (n_embd, head_size) to (n_embd, n_embd)
        self.Wk = nn.Linear(n_embd, n_embd, bias=False) # (512, 64) 
        self.Wq = nn.Linear(n_embd, n_embd, bias=False) # (512, 64) 
        self.Wv = nn.Linear(n_embd, n_embd, bias=False) # (512, 64)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
        # residual connections and dropout for efficiency
        self.skip_connection = nn.Linear(n_embd, n_embd) # (64, 512) # changed (head_size,n_embd) to (n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # self-attention
        #print(f"x shape inside MultiHead: {x.shape}") # (32,10,512) #(DEBUG)
        #(DEBUG)print(f"Wk shape: {self.Wk}")
        B,T,C = x.shape # B:32, T:10, C:512 #B:64, T:256, C:384
        k = self.Wk(x) # (32,10,512) @ (512, 64) -> (32,10,64)
        q = self.Wq(x) # (32,10,512) @ (512, 64) -> (32,10,64)
        v = self.Wv(x) # (32,10,512) @ (512, 64) -> (32,10,64) 
        #print(f"k.shape {k.shape}, q.shape {q.shape}, v.shape {v.shape}") # (32,10,64) #(DEBUG)
        
        # splitting k,q,v into their heads changing their shapes
        k = k.view(B, T, n_heads, C//n_heads).transpose(-3, -2) # (32,8,10,8)
        q = q.view(B, T, n_heads, C//n_heads).transpose(-3, -2) # (32,10,64) -> (32,10,8,64/8)
        v = v.view(B, T, n_heads, C//n_heads).transpose(-3, -2)
        #(DEBUG)print(f"k shape after view and transpose {k.shape}")
        #(DEBUG)print(f"k.shape after VIEW: {k.view(B, T, n_heads, k.shape[-1]//n_heads).shape}") # (32,10,8,64)
        #(DEBUG)print(f"k.shape after TRANSPOSE: {k.view(B, T, n_heads, k.shape[-1]//n_heads).transpose(-3, -2).shape}") # (32,10,8,64) 

        # (32,8,10,8) @ (32,8,8,10) -> (32,8,10,10) porque (10,8) @ (8,10) = (10,10)
        # (B,n_heads,T,q.shape[-1]) @ (B,n_heads,k.shape[-1],T) -> (B,n_heads,T,T)
        affinities = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5) # C**-0.5 = math.sqrt(C)
        affinities = affinities.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        #affinities = self.dropout(affinities)

        # weight aggregation with values
        aggregation = affinities @ v # (32,8,10,10) @ (32,8,10,8) -> (32,8,10,8)
        #(DEBUG)print("aggregation:", aggregation.shape)

        # reshape back 
        # erro aqui (32,8,10,8).transpose(-3,-2) -> (32,10,8,8) -> .view(B,-1,n_heads*)
        aggregation = aggregation.transpose(-3, -2)              # (32,10,8,8)
        aggregation = aggregation.reshape(B, T, n_heads*(C//n_heads)) # (32,10,8*8)
        #(DEBUG) print("aggregation:", aggregation.shape)

        # residual connection and dropout
        aggregation = self.skip_connection(aggregation) # (32,10,64) @ (64,512) -> (32,10,512)
        aggregation = self.dropout(aggregation)
        #print("aggregation", aggregation.shape) # 64, 256, 384
        return aggregation

# Feed Forward class
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), #(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out

# communication and computation
class TransformerBlock(nn.Module):

    def __init__(self, n_embd, n_heads): # 512, 8
        super().__init__()
        head_size = n_embd // n_heads # 512/8 = 64
        #(DEBUG)print(f"n_embd:{n_embd}, n_heads:{n_heads}, head_size:{head_size}")
        #self.self_attention = MultiHeadAttention(n_heads, head_size)
        self.attention = MultiHeadAttention_attempt(head_size, n_heads) # (64, 8)
        self.feedforward = FeedForward(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #residual = x 
        #(DEBUG)print(f"x shape inside TransformerBlock: {x.shape}") # (32,10,512)
        out = x + self.attention(self.layernorm1(x))
        out = x + self.feedforward(self.layernorm2(x))
        return out

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        # token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)      # (65, 512): each token has a representation of a vector of 512 values
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # (10, 512): each index of the block will have also a positional representation of 512 values
        # transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_heads) for i in range(n_layers)]) # (512, 8)
        self.final_layernorm = nn.LayerNorm(n_embd)
        self.softmax = nn.Linear(n_embd, vocab_size)
        # The sequential will be: nn.Sequential(TransformerBlock(), TransformerBlock(),..., TransformerBlock())
        # better init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B,T = index.shape
        token_embeddings = self.token_embedding_table(index) # (B,T,C)
        positional_embeddings = self.positional_embedding_table(torch.arange(T, device=device)) # T: block_size
        emb = token_embeddings + positional_embeddings # (B,T,C)
        x = self.blocks(emb) # (B,T,C)
        x = self.final_layernorm(x) # (B,T,C)
        logits = self.softmax(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for i in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, next), dim=1)
        return index       
        
model = GPT()
m = model.to(device)

# Number of parameters in the model
print(sum(p.numel() for p in m.parameters()),'params')#/1e6, 'M parameters')

# Adam optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iterations):
    # every once in a while evaluate the loss
    if iter % eval_interval == 0 or iter == max_iterations-1:
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))