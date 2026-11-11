import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers, 
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ViT & CrossViT
# Attention class for multi-head self-attention mechanism with softmax and dropout
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        # set heads and scale (=sqrt(dim_head))
        self.heads = heads
        self.scale = dim_head ** -0.5

        total_dim = heads * dim_head

        # we need softmax layer and dropout
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(
            dropout)  # randomly sets some attention weights to zero during training to reduce overfitting

        # as well as the q linear layer
        self.to_qkv = nn.Linear(dim, total_dim * 3, bias=False)

        # and the k/v linear layer (can be realized as one single linear layer
        # or as two individual ones)
        # and the output linear layer followed by dropout
        self.to_out = nn.Sequential(
            nn.Linear(total_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        # now compute the attention/cross-attention
        # in cross attention: x = class token, context = token embeddings
        # don't forget the dropout after the attention
        # and before the multiplication w. 'v'
        # the output should be in the shape 'b n (h d)'
        b, n, _, h = *x.shape, self.heads

        if context is None:
            context = x

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        # Queries (q) come from x (e.g., [CLS] token)
        q = self.to_qkv(x).chunk(3, dim=-1)[0]
        # Keys (k) and Values (v) come from context (e.g., patch tokens of the other branch)
        k, v = self.to_qkv(context).chunk(3, dim=-1)[1:]

        q = q.reshape(b, n, h, -1).transpose(1, 2)
        k = k.reshape(b, k.shape[1], h, -1).transpose(1, 2)
        v = v.reshape(b, v.shape[1], h, -1).transpose(1, 2)

        dot_attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        self_attention = self.attend(dot_attention)  # sm
        self_attention = self.dropout(self_attention)

        out = torch.matmul(self_attention, v)  # apply weights
        out = out.transpose(1, 2).reshape(b, n, -1)  # combine heads together

        return self.to_out(out)


# ViT & CrossViT
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# CrossViT
# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
    """
    Adapter class that embeds a callable (layer) and handles mismatching dimensions
    """

    def __init__(self, dim_outer, dim_inner, fn):
        """
        Args:
            dim_outer (int): Input (and output) dimension.
            dim_inner (int): Intermediate dimension (expected by fn).
            fn (callable): A callable object (like a layer).
        """
        super().__init__()
        self.fn = fn
        need_projection = dim_outer != dim_inner
        self.project_in = nn.Linear(dim_outer, dim_inner) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_inner, dim_outer) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        """
        Args:
            *args, **kwargs: to be passed on to fn

        Notes:
            - after calling fn, the tensor has to be projected back into it's original shape   
            - fn(W_in) * W_out
        """

        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


# CrossViT
# cross attention transformer
class CrossTransformer(nn.Module):
    # This is a special transformer block
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        inner_dim = dim_head * heads
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, inner_dim,
                             PreNorm(inner_dim, Attention(inner_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(lg_dim, inner_dim,
                             PreNorm(inner_dim, Attention(inner_dim, heads=heads, dim_head=dim_head, dropout=dropout)))
            ]))
        # Note: no positional FFN here 

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]),
                                                                   (sm_tokens, lg_tokens))

        # Forward pass through the layers, 
        # cross attend to 
        # 1. small cls token to large patches and
        # 2. large cls token to small patches
        for sm_attn, lg_attn in self.layers:
            # 1. Small CLS attends to Large patches
            sm_cls = sm_cls + sm_attn(sm_cls, context=lg_patch_tokens, kv_include_self=True)

            # 2. Large CLS attends to Small patches
            lg_cls = lg_cls + lg_attn(lg_cls, context=sm_patch_tokens, kv_include_self=True)

        # finally concat sm/lg cls tokens with patch tokens
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens


# CrossViT
# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
            self,
            *,
            depth,
            sm_dim,
            lg_dim,
            sm_enc_params,
            lg_enc_params,
            cross_attn_heads,
            cross_attn_depth,
            cross_attn_dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=sm_dim, **sm_enc_params, dropout=dropout),
                # 2. Large branch transformer encoder
                Transformer(dim=lg_dim, **lg_enc_params, dropout=dropout),
                # 3. Cross-transformer block for communication
                CrossTransformer(
                    sm_dim=sm_dim,
                    lg_dim=lg_dim,
                    depth=cross_attn_depth,
                    heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head,
                    dropout=dropout
                )
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attn in self.layers:
            # Independent self-attention refinement
            sm_tokens = sm_enc(sm_tokens)
            lg_tokens = lg_enc(lg_tokens)

            # Cross-scale communication (sharing information between [CLS] tokens and patches)
            sm_tokens, lg_tokens = cross_attn(sm_tokens, lg_tokens)
        return sm_tokens, lg_tokens


# CrossViT (could actually also be used in ViT)
# helper function that makes the embedding from patches
# have a look at the image embedding in ViT
class ImageEmbedder(nn.Module):
    def __init__(
            self,
            *,
            dim,
            image_size,
            patch_size,
            dropout=0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # create layer that re-arranges the image patches
        # and embeds them with layer norm + linear projection + layer norm
        self.to_patch_embedding = nn.Sequential(
            # taken directly from the VIT
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # create/initialize #dim-dimensional positional embedding (will be learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # create #dim cls tokens (for each patch embedding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # create dropput layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # forward through patch embedding layer
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # concat class tokens
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        return self.dropout(x)


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding 
        # or the class token
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        # to a latent space, which can then be used as input
        # to the mlp head
        x = self.to_latent(x)
        return self.mlp_head(x)


# CrossViT
class CrossViT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            num_classes,
            sm_dim,
            lg_dim,
            sm_patch_size=12,
            sm_enc_depth=1,
            sm_enc_heads=8,
            sm_enc_mlp_dim=2048,
            sm_enc_dim_head=64,
            lg_patch_size=16,
            lg_enc_depth=4,
            lg_enc_heads=8,
            lg_enc_mlp_dim=2048,
            lg_enc_dim_head=64,
            cross_attn_depth=2,
            cross_attn_heads=8,
            cross_attn_dim_head=64,
            depth=3,
            dropout=0.1,
            emb_dropout=0.1
    ):
        super().__init__()
        # create ImageEmbedder for small and large patches
        self.small_embedder = ImageEmbedder(dim=sm_dim, image_size=image_size,
                                            patch_size=sm_patch_size, dropout=dropout)
        self.large_embedder = ImageEmbedder(dim=lg_dim, patch_size=lg_patch_size,
                                            image_size=image_size, dropout=dropout)
        # create MultiScaleEncoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim,
                dim_head=sm_enc_dim_head
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim,
                dim_head=lg_enc_dim_head
            ),
            dropout=dropout
        )

        # create mlp heads for small and large patches
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, img):
        # apply image embedders
        sm_embed = self.small_embedder.forward(img)
        lg_embed = self.large_embedder.forward(img)
        # and the multi-scale encoder
        sm_tokens, lg_tokens = self.multi_scale_encoder.forward(sm_embed, lg_embed)
        # call the mlp heads w. the class tokens
        sm_cls = sm_tokens[:, 0]
        lg_cls = lg_tokens[:, 0]
        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)
        return sm_logits + lg_logits


if __name__ == "__main__":
    x = torch.randn(16, 3, 32, 32)

    #vit = ViT(image_size=32, patch_size=8, num_classes=10, dim=64, depth=2, heads=8, mlp_dim=128, dropout=0.1,
    #          emb_dropout=0.1)

    cvit = CrossViT(image_size=32, num_classes=10, sm_dim=64, lg_dim=128, sm_patch_size=8,
                    sm_enc_depth=2, sm_enc_heads=8, sm_enc_mlp_dim=128, sm_enc_dim_head=64,
                    lg_patch_size=16, lg_enc_depth=2, lg_enc_heads=8, lg_enc_mlp_dim=128,
                    lg_enc_dim_head=64, cross_attn_depth=2, cross_attn_heads=8, cross_attn_dim_head=64,
                    depth=3, dropout=0.1, emb_dropout=0.1)

    #print(vit(x).shape)

    print(cvit(x).shape)
