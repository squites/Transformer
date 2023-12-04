import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64#32 #4
block_size = 256 #10 #8 # sequence length
n_embd = 384#512 #32
n_heads = 6#8 #4 #6
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

# trying to make multi-head class without having to do a self-attention class, so treating heads as a dim
class MaskedMultiheadAttention(nn.Module):
    """
        Computes MultiHead self-attention in parallel, treating heads as a dimension
    """
    def __init__(self, n_heads): # 32, 8 # There was head_size as a parameter too
        super().__init__()
        assert n_embd % n_heads == 0 # 512%8 == 0 -> True
        #self.heads = n_heads # 8
        #(DEBUG)print(f"n_embd, head_size: {n_embd}, {head_size}")
        self.Wk = nn.Linear(n_embd, n_embd, bias=False) # (384, 384) 
        self.Wq = nn.Linear(n_embd, n_embd, bias=False) # (384, 384) 
        self.Wv = nn.Linear(n_embd, n_embd, bias=False) # (384, 384)
        self.out_linear = nn.Linear(n_embd, n_embd, bias=False) 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
        # residual connections and dropout for efficiency
        self.skip_connection = nn.Linear(n_embd, n_embd) # (384, 384)  # changed (head_size,n_embd) to (n_embd,n_embd)
        self.affinities_drop = nn.Dropout(dropout)
        self.residual_drop = nn.Dropout(dropout)

    def forward(self, x):
        # self-attention
        #print(f"x shape inside MultiHead: {x.shape}") # (32,10,512) #(DEBUG)
        #(DEBUG)print(f"Wk shape: {self.Wk}")
        B,T,C = x.shape # B:64, T:256, C:384 
        # These are the actual vectors
        k = self.Wk(x) # (64,256,384) @ (384, 384) -> (64,256,384)
        q = self.Wq(x) # (64,256,384) @ (384, 384) -> (64,256,384)
        v = self.Wv(x) # (64,256,384) @ (384, 384) -> (64,256,384)
        #print(f"k.shape {k.shape}, q.shape {q.shape}, v.shape {v.shape}") # (64,256,384) #(DEBUG)
        
        # splitting k,q,v into their heads changing their shapes
        k = k.view(B, T, n_heads, C//n_heads).transpose(-3, -2) # (64,6,256,64) # 384/6=64
        q = q.view(B, T, n_heads, C//n_heads).transpose(-3, -2) # permute?
        v = v.view(B, T, n_heads, C//n_heads).transpose(-3, -2)
        #(DEBUG)print(f"k shape after view and transpose {k.shape}")
        #(DEBUG)print(f"k.shape after VIEW: {k.view(B, T, n_heads, k.shape[-1]//n_heads).shape}") # (32,10,8,64)
        #(DEBUG)print(f"k.shape after TRANSPOSE: {k.view(B, T, n_heads, k.shape[-1]//n_heads).transpose(-3, -2).shape}") # (32,10,8,64) 

        # Scaled Dot-Product attention -----
        # (64,6,256,64) @ (64,6,64,256) -> (64,6,256,256) porque (10,8) @ (8,10) = (10,10)
        # (B,n_heads,T,q.shape[-1]) @ (B,n_heads,k.shape[-1],T) -> (B,n_heads,T,T)
        affinities = q @ k.transpose(-2, -1) * math.sqrt(k.shape[-1]) #(k.shape[-1]**-0.5)
        affinities = affinities.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        affinities = self.affinities_drop(affinities)

        # weight aggregation with values # Is this the Linear layer that is on the diagram after Scaled Dot-product attention?
        aggregation = affinities @ v # (64,6,256,256) @ (64,6,256,64) -> (64,6,256,64)
        #(DEBUG)print("aggregation:", aggregation.shape)                  0  2  1  3
        # -----

        # reshape back # note: I'll try with permute().contiguous() following the blog, maybe this is the error
        aggregation = aggregation.transpose(-3, -2).contiguous().view(B,T,C) # (64,256,6,64)
        #aggregation = aggregation.permute(0, 2, 1, 3).contiguous()
        #aggregation = aggregation.view(B, T, C)  # (64,256,384)
        #aggregation = aggregation.reshape(B,T,aggregation[-2]*aggregation[-1]) # it says to reshape() here instead of view()
        #(DEBUG) print("aggregation:", aggregation.shape)

        # residual connection and dropout
        aggregation = self.skip_connection(aggregation)
        #aggregation = self.residual_drop(aggregation)
        #print("aggregation", aggregation.shape) # 64, 256, 384
        aggregation = self.out_linear(aggregation)
        return aggregation # (64,256,384)

# MLP class: computation
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # (384, 1536)
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # (1536, 384)
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x) # (64,256,384) @ (384,1536) -> (64,256,1536) @ (1536,384) -> (64,256,384)
        return out

# communication and computation
class TransformerBlock(nn.Module):

    def __init__(self, n_embd, n_heads): # 384, 6 
        super().__init__()
        #head_size = n_embd // n_heads # 512/8 = 64
        #(DEBUG)print(f"n_embd:{n_embd}, n_heads:{n_heads}, head_size:{head_size}")
        self.attention = MaskedMultiheadAttention(n_heads) # (6)
        self.feedforward = FeedForward(n_embd) # 384
        self.layernorm1 = nn.LayerNorm(n_embd) # this goes before attention
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #(DEBUG)print(f"x shape inside TransformerBlock: {x.shape}") # (32,10,512)
        out = x + self.attention(self.layernorm1(x))
        out = x + self.feedforward(self.layernorm2(x))
        return out

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        # token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)      # (65, 384): each token has a representation of a vector of 384 values
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # (10, 384): each index of the block will have also a positional representation of 384 values
        # transformer blocks
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(n_embd, n_heads) for i in range(n_layers)]) # will execute sequentially n TransformerBlocks
        self.final_layernorm = nn.LayerNorm(n_embd)
        # shouldn't have another linear layer here, right before the softmax?
        self.softmax = nn.Linear(n_embd, vocab_size)
        # The sequential will be: nn.Sequential(TransformerBlock(), TransformerBlock(),..., TransformerBlock()). Executes sequentially everything before moving on to something different
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
        token_embeddings = self.token_embedding_table(index) # (65, 384)[index]: meaning takes the index row in the table (65,384)
        positional_embeddings = self.positional_embedding_table(torch.arange(T, device=device)) # T: block_size
        emb = token_embeddings + positional_embeddings # (1,384) + (1,384) -> (1,384) # for 1 index
        x = self.transformer_blocks(emb) # (1,384)
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
print(sum(p.numel() for p in m.parameters())/1e6,'M parameters')#/1e6, 'M parameters')

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

# desired loss values: train 1.0763, val 1.4873
# note: For every word, there's a query, key and value vector. In our case, we're using a single token for
# a single character, so for every character(token) we'll have a query,key,value vector.