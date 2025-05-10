import jax
import jax.numpy as jnp
from flax import linen as nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.hidden_dim),
            nn.gelu,  
            nn.Dense(self.dim)
        ])(x)
        return x


class Attention(nn.Module):
    dim: int
    heads: int = 8
    dim_head: int = 64

    def setup(self):
        self.inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head ** -0.5

        self.norm = nn.LayerNorm()
        self.to_qkv = nn.Dense(self.inner_dim * 3, use_bias=False)
        self.to_out = nn.Dense(self.dim, use_bias=False)

    def __call__(self, x):
        x = self.norm(x)
        
        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)  

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = jnp.matmul(q, k.swapaxes(-1, -2)) * self.scale
        attn = nn.softmax(dots, axis=-1)  

        out = jnp.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int
    mlp_dim: int
    use_bias: bool = True
    
    def setup(self):
        self.norm = nn.LayerNorm(axis=-1)
        self.layers = [self._make_layer() for _ in range(self.depth)]
    
    def _make_layer(self):
        return nn.ModuleList([
            Attention(self.dim, self.heads, self.dim_head, self.use_bias),
            FeedForward(self.dim, self.mlp_dim, self.use_bias)
        ])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for attn, ff in self.layers:
            x = attn(x) + x  
            x = ff(x) + x    
        return self.norm(x)


class SimpleViT(nn.Module):
    image_size: int 
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    channels: int = 3
    dim_head: int = 64

    def setup(self):
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        assert image_height % patch_height == 0, "图像高度必须能被块高度整除"
        assert image_width % patch_width == 0, "图像宽度必须能被块宽度整除"
        
        self.to_patch_embedding = nn.Sequential([
            lambda x: rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                              p1=patch_height, p2=patch_width),
            nn.LayerNorm(),
            nn.Dense(self.dim),
            nn.LayerNorm()
        ])
        
        self.pos_embedding = self.param('pos_embed', 
                                      nn.initializers.normal(stddev=0.02),
                                      (1, (image_height//patch_height)*(image_width//patch_width)+1, self.dim))
        
        self.transformer = Transformer(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim
        )
        self.to_latent = nn.Identity()
        
        self.linear_head = nn.Dense(self.num_classes)

    def __call__(self, img, train: bool = True):
        x = self.to_patch_embedding(img)
        
        cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.dim))
        cls_tokens = jnp.tile(cls_token, (img.shape[0], 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += self.pos_embedding
        
        x = self.transformer(x)
        
        x = x[:, 0]
        
        return self.linear_head(x)