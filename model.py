import math
import inspect
import tiktoken
import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.dropout = dropout
        self.n_head = n_head
        self.block_size = block_size
        self.ln_1 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(self.n_embd, self.n_head, self.dropout,self.block_size )
        self.ln_2 = nn.LayerNorm(self.n_embd)
        self.mlp = MLP(n_embd=self.n_embd, dropout=self.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.block_size = block_size
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                        .view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # Cut the qkv in the "vertical" dim (Channel dim, output of embeding) in order to process it in different heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y






class MLP(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.dropout = dropout

        self.c_fc    = nn.Linear(self.n_embd, 1 * self.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(1 * self.n_embd, self.n_embd, bias=False)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    



class GPT(nn.Module):

    def __init__(self,
                 vocab_size,
                 block_size,
                 n_layer=12,
                 n_head=12,
                 n_embd=768,
                 dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.n_embd),
            wpe = nn.Embedding(self.block_size, self.n_embd),
            drop = nn.Dropout(self.dropout),
            h = nn.ModuleList([Block(self.n_embd, self.n_head, self.dropout,self.block_size) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        # self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
            # if pn.endswith('c_proj.weight'):
                # torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layer))


    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def generate(self, input_tokens : torch.LongTensor, max_length) -> torch.LongTensor :
        """Generate a sequence of tokens"""

        for _ in range(max_length):
            # clip the input ids to block size
            input_tokens = input_tokens[:, -self.block_size:]
            logits, loss = self(input_tokens)
            # take the logits of the last position and apply softmax to get log probs
            # log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)

            # sample from the distribution or take the most likely
            next_token = torch.multinomial(probs, num_samples=1)

            # concatenate the sampled token to the input
            input_tokens = torch.cat((input_tokens, next_token), dim=1)


        return input_tokens