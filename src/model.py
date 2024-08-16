
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack


# https://github.com/insitro/ChannelViT
class PatchEmbedPerChannel(nn.Module):

    def __init__(
        self,
        sig_size: int=3000,
        patch_size: int=100,
        in_chans: int=4,
        embed_dim: int=512,
    ):
        super().__init__()
        num_patches = (sig_size // patch_size) * in_chans
        self.sig_size = sig_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size),
        )

        self.channel_embed = nn.parameter.Parameter(
                    torch.zeros(1, embed_dim, in_chans, 1)
                )
        # trunc_normal_(self.channel_embed, std=0.02)

    def forward(self, x):

        B, Cin, S = x.shape

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin S
        # channel specific offsets
        x += self.channel_embed[:, :, :, :]  # B Cout Cin S

        # preparing the output sequence
        x = x.flatten(2)  # B Cout CinS
        x = x.transpose(1, 2)  # B CinS Cout

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, num_patches, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Backbone(nn.Module):
    def __init__(self, *, seq_len, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = (seq_len // patch_size) * channels
        self.to_patch_embedding = PatchEmbedPerChannel(
            sig_size=seq_len, 
            patch_size=patch_size, 
            in_chans=channels, 
            embed_dim=dim
            )

        self.cls_token = nn.Parameter(torch.randn(dim))

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.num_extra_tokens = 1  # cls token
        # self.num_extra_tokens = 0  # avg pooling
        self.pos_embedding = nn.Parameter(
            # torch.zeros(1, num_patches // channels + self.num_extra_tokens, dim)
            torch.randn(1, num_patches // channels + self.num_extra_tokens, dim)
            )
        # trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout)

    def interpolate_pos_encoding(self, x, L, c):

        # number of auxiliary dimensions before the patches
        if not hasattr(self, "num_extra_tokens"):
            # backward compatibility
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.num_extra_tokens

        npatch = x.shape[1] - num_extra_tokens
        N = self.pos_embedding.shape[1] - num_extra_tokens

        if npatch == N:
            return self.pos_embedding

        class_pos_embed = self.pos_embedding[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embedding[:, num_extra_tokens:]

        dim = x.shape[-1]
        L0 = L // self.to_patch_embedding.patch_size

        # see discussion at https://github.com/facebookresearch/dino/issues/8
        L0 += 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, N, dim).permute(0, 2, 1),
            size=int(L0),
            mode="linear",
        ) # torch.Size([1, 512, 30])

        assert int(L0) == patch_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.permute(0, 2, 1).view(1, 1, -1, dim)
        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(-1, c, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


    def forward(self, x):

        b, c, L = x.shape

        x = self.to_patch_embedding(x)
        _, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')

        # x += self.pos_embedding[:, :(n+1)]
        x = x + self.interpolate_pos_encoding(x, L, c) # torch.Size([512, 121, 512]) + torch.Size([1, 121, 512])

        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')
        return cls_tokens
        # return torch.mean(x, dim=1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.net1 = Backbone(
                seq_len=3000,
                channels=4,
                patch_size=100,
                dim=512,
                depth=6,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
        )

        self.nerm_ff = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
        )
        self.nerm_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 2)           
        )
        self.depth_ff = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
        )
        self.depth_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 1),
        )

    def forward(self, psg):
        psg_feat = self.net1(psg)

        nerm_feat = self.nerm_ff(psg_feat) + psg_feat
        depth_feat = self.depth_ff(psg_feat) + psg_feat

        return self.depth_head(depth_feat), self.nerm_head(nerm_feat)