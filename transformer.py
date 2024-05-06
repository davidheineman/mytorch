import torch
from torch.nn import Module, ModuleList, Embedding
from torch.utils.data import Dataset

import nn
import functional as F

from utils import CfgNode as CN
from tqdm import tqdm


class Embedding(Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_embeddings = Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = Embedding(config.block_size, config.n_embd)
    
    def forward(self, idx):
        B, T = idx.size()
        
        positions  = torch.ones(B, dtype=torch.long)[:, None] @ torch.arange(T)[None, :]
        embeddings = self.vocab_embeddings(idx) + self.position_embeddings(positions)
        
        return embeddings


class GenericSelfAttention(Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Note: These could be a single batched linear layer
        # but we separate them for simplicity of implementation.
        self.k = nn.Linear(config.n_embd, config.n_embd)
        self.v = nn.Linear(config.n_embd, config.n_embd)
        self.q = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.hidden_dropout = nn.Dropout(config.hidden_pdrop)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attention_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        D = C // self.n_head

        q, k, v = self.q(x), self.k(x), self.v(x)
        k = k.reshape(B, T, self.n_head, D).transpose(1, 2) # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_head, D).transpose(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, D).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn = (q @ k.transpose(-2, -1)) / (D ** 0.5)
        attn = attn.masked_fill(attention_mask[:,:,:T] == 0, float('-inf'))
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        y = (attn @ v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(B, T, C)

        y = self.hidden_dropout(self.c_proj(y))

        return y
    

class TransformerBlock(Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GenericSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = torch.nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=nn.GELU(),
                dropout=nn.Dropout(config.hidden_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    # A Basic Transformer Block with Attention followed by an MLP
    # note the layer norms and residual information preserved at each step.
    def forward(self, x, attention_mask):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class GenericTransformer(Module):
    @staticmethod
    def get_default_config():
        C = CN()
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.hidden_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])

        self.block_size = config.block_size
        self.transformer = torch.nn.ModuleDict(dict(
            embedding=Embedding(config),
            h=ModuleList(
                [TransformerBlock(config) for _ in range(config.n_layer)]
            ),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

        self.block_size = config.block_size # Maximum Number of Tokens which can be encoded at once
        self.vocab_size = config.vocab_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params, union_params = decay & no_decay, decay | no_decay
        assert (len(inter_params) == 0), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (len(param_dict.keys() - union_params) == 0), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!" 

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        return optim_groups

    def get_attention_mask(self, num_tokens):
        raise NotImplementedError()

    def forward(self, idx, targets=None, hidden_cache=None, return_hidden=False):
        """
        :param idx: int Tensor of shape (B,T)
        :param hidden_cache: float Tensor of shape (B,P_T,n_embd)
        :param targets: int Tensor of shape (B,T_T)
        :param return_hidden: bool
        (if return_hidden = None)
        :returns x: float Tensor of shape (B,T,n_embd)
        (else)
        :returns logits: float Tensor of shape (B, T, vocab_size)
        :returns loss: float Tensor of shape (B) or None
        """
        num_tokens = (idx != -1).type(torch.int).sum(dim=1)
        if hidden_cache is not None:
          num_tokens = num_tokens + hidden_cache.shape[1]
        idx = idx.masked_fill(idx == -1, int(0)).type(torch.int)[:, :num_tokens.max().item()]
        if targets is not None:
          targets = targets[:, :num_tokens.max().item()]
        attention_mask = self.get_attention_mask(num_tokens)

        ###### MAIN COMPUTATION ######
        x = self.transformer.embedding(idx)
        if hidden_cache is not None:
          x = torch.cat((hidden_cache, x), dim=1)
        for block in self.transformer.h:
          x = block(x, attention_mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        ###### MAIN COMPUTATION ######

        if return_hidden: return x

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            s_logits = logits[:, hidden_cache.shape[1]-1:-1].contiguous() if hidden_cache is not None else logits
            loss = F.ce(s_logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-1)

        return logits, loss


class Encoder(GenericTransformer):
    """Encoder Style Transformer with Bidirectional Attention"""
    def get_attention_mask(self, num_tokens):
        B = num_tokens.shape[0]
        max_tokens = min(self.block_size, num_tokens.max().item())

        attention_mask = torch.zeros(B, 1, max_tokens, max_tokens)
        for b in range(B):
            attention_mask[b, 0, :, :num_tokens[b]] = 1

        return attention_mask.reshape(B, 1, max_tokens, max_tokens)


class Decoder(Encoder):
    """Decoder Style model with a Causal Attention Mask"""
    def get_attention_mask(self, num_tokens):
        full_attention_mask = super().get_attention_mask(num_tokens)

        attention_mask = full_attention_mask * torch.tril(torch.ones_like(full_attention_mask), diagonal=0)
        
        return attention_mask


class EncoderDecoder(Module):
    """Encoder-Decoder Model which combines the two architectures"""
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        # Add end of sequence token.
        decoder_config.vocab_size += 1
        self.vocab_size = decoder_config.vocab_size
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def configure_optimizers(self, train_config):
        enc_groups = self.encoder.configure_optimizers(train_config)
        dec_groups = self.decoder.configure_optimizers(train_config)
        return enc_groups + dec_groups

    def forward(self, prefix, targets=None):
        """
        :param prefix: int Tensor of shape (B,P_T)
        :param idx: float Tensor of shape (B,P_T,n_embd)
        :returns logits: float Tensor of shape (B, vocab_size)
        :returns loss: float Tensor of shape (B) or None
        """
        B = prefix.shape[0]
        idx = torch.tensor([[]]).repeat(B, 1)
        if targets is not None:
          idx = torch.cat((idx, targets), dim=1)
          
        h = self.encoder.forward(prefix, targets=None, hidden_cache=None, return_hidden=True)
        logits, loss = self.decoder.forward(idx, targets=targets, hidden_cache=h, return_hidden=False)
        
        return logits, loss


def generate(model, idx, max_new_tokens, temperature=1.0):
    """
    :param idx: int Tensor of shape (B, T)
    :param max_new_tokens: int
    :param temperature: Float
    :returns idx: int Tensor of shape (B, T+max_new_tokens)
    """
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        probs = F.softmax(logits / temperature, dim=-1)
        next_toks = torch.multinomial(probs[:, -1], 1)
        idx = torch.cat([idx, next_toks], dim=1)
    return idx


def prefix_generate(model, prefix, max_new_tokens, temperature=1.0):
    """
    :param prefix: int Tensor of shape (B, T)
    :param max_new_tokens: int
    :param temperature: Float
    :returns idx: int Tensor of shape (B, max_new_tokens)
    """
    idx = torch.tensor([[]], dtype=torch.long)

    for _ in range(max_new_tokens):
        logits, _ = model(torch.cat([prefix, idx], dim=1))
        probs = F.softmax(logits / temperature, dim=-1)
        next_toks = torch.multinomial(probs[:, -1], 1)
        idx = torch.cat([idx, next_toks], dim=1)

    return idx


class LMDataset(Dataset):
    def __init__(self, split, data, tokenizer, model):
        assert split in {'train', 'test'}
        self.model_type = "EncDec" if issubclass(type(model), EncoderDecoder) else "Dec"
        if split == "train":
          self.start_split = 0
          self.end_split = 30000
        else:
          self.start_split = 30000
          self.end_split = 40000
        self.split = split
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = max([len(self.tokenizer.encode(inp)) for inp in self.data])
        self.process()

    def __len__(self):
        return len(self.data[self.start_split:self.end_split])

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.block_size

    def process(self):
      new_data = []
      for inp in tqdm(self.data):
        if self.model_type == "EncDec":
          x_inp = inp.split("[SEP]")[0] + "[SEP]"
          y_inp = inp.split("[SEP]")[1]
          x = self.tokenizer.encode(x_inp)
          y = self.tokenizer.encode(y_inp)
        else:
          x = self.tokenizer.encode(inp)
          y = x[1:]
          x = x[:-1]
        x = x + ([-1] * (self.get_block_size() - len(x)))
        y = y + ([-1] * (self.get_block_size() - len(y)))
        new_data.append((x, y))
      self.data = new_data

    def __getitem__(self, idx):
      x, y = self.data[self.start_split + idx]
      return torch.tensor(x), torch.tensor(y)


class Tokenizer():
    """
    Incredibly Simplified Tokenizer so that you can manually hack it!
    """
    def __init__(self):
        self.DELIM = "|[DELIM]|"
        self.special_tokens = ["[SEP]", "[END]"]
        self.special_tokens = [self.stringify(list(bytes(tok, "utf-8"))) for tok in self.special_tokens]
        self.vocab_size = 256 + len(self.special_tokens)

    def stringify(self, b_enc):
        s_enc = [str(b) for b in b_enc]
        return self.DELIM.join(s_enc)

    def get_vocab_size(self):
        return self.vocab_size

    def encode(self, inp):
        s_enc = self.stringify(list(bytes(inp, "utf-8")))
        for i, tok in enumerate(self.special_tokens):
            s_enc = s_enc.replace(tok, str(255+i+1))
        return [int(s) for s in s_enc.split(self.DELIM)]

    def decode(self, inp):
        s_enc = self.stringify(inp)
        for i, tok in enumerate(self.special_tokens):
            s_enc = s_enc.replace(str(255+i+1), tok)
        return  bytes([int(c) for c in s_enc.split(self.DELIM)])