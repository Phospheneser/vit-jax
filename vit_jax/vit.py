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
    use_bias: bool = False   # 新增偏置控制
    
    @nn.compact
    def __call__(self, x):
        # 使用 Flax 的 Sequential 替代 PyTorch 的 Sequential
        # 注意 JAX 的 LayerNorm 默认对最后一个轴归一化
        x = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.hidden_dim, use_bias=self.use_bias),
            nn.gelu,  # 直接使用函数式激活函数
            nn.Dense(self.dim, use_bias=self.use_bias)
        ])(x)
        return x


class Attention(nn.Module):
    dim: int           # 输入维度
    heads: int = 8     # 总查询头数
    groups: int = 2    # 分组数量
    dim_head: int = 64
    use_bias: bool = False   # 新增qkv偏置控制
    use_qk_norm: bool = True # 新增QK归一化控制
    qk_norm_eps: float = 1e-6
    qk_norm_type: str = 'layer' # 支持layer/rms两种归一化
    
    def setup(self):
        assert self.heads % self.groups == 0, "head dim must be divisible by groups, but got head num:{} and group num:{}".format(self.heads, self.groups)
        self.inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head ** -0.5

        # 核心参数初始化（参考Flax文档）
        kernel_init = nn.initializers.lecun_normal()
        bias_init = nn.initializers.zeros
        
        # 投影层定义（增加use_bias参数）
        self.norm = nn.LayerNorm()
        self.to_q = nn.Dense(
            self.inner_dim, 
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init
        )
        self.to_kv = nn.Dense(
            2 * self.dim_head * self.groups,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init
        )
        
        # QK归一化层（根据类型选择）
        if self.use_qk_norm:
            if self.qk_norm_type == 'layer':
                self.q_norm = nn.LayerNorm(epsilon=self.qk_norm_eps)
                self.k_norm = nn.LayerNorm(epsilon=self.qk_norm_eps)
            elif self.qk_norm_type == 'rms':
                self.q_norm = RMSNorm(epsilon=self.qk_norm_eps)
                self.k_norm = RMSNorm(epsilon=self.qk_norm_eps)
        
        self.to_out = nn.Dense(self.dim, use_bias=False)

    def __call__(self, x):
        x = self.norm(x)
        B, N, _ = x.shape
        
        # 生成查询向量
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        
        # 生成分组键值向量
        kv = self.to_kv(x)
        k, v = jnp.split(kv, 2, axis=-1)
        k = rearrange(k, 'b n (g d) -> b g n d', g=self.groups)
        v = rearrange(v, 'b n (g d) -> b g n d', g=self.groups)
        
        # 应用QK归一化（参考LLaMA设计）
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # 计算分组注意力
        q_groups = rearrange(q, 'b (g h) n d -> b g h n d', g=self.groups)
        dots = jnp.einsum('bghqd,bgkd->bghqk', q_groups, k) * self.scale
        attn = nn.softmax(dots, axis=-1)
        
        # 分组加权求和
        out = jnp.einsum('bghqk,bgkd->bghqd', attn, v)
        out = rearrange(out, 'b g h n d -> b n (g h d)')
        
        return self.to_out(out)

class RMSNorm(nn.Module):
    # 新增RMS归一化层（兼容Flax模块设计）
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        return x * self.param('scale', nn.initializers.ones, (x.shape[-1],))





class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    groups: int
    dim_head: int
    mlp_dim: int
    use_bias: bool = True
    
    def setup(self):
        self.norm = nn.LayerNorm()
        self.layers = [self._make_layer() for _ in range(self.depth)]
    
    def _make_layer(self):
        return (
            Attention(self.dim, self.heads, self.groups, self.dim_head, self.use_bias),
            FeedForward(self.dim, self.mlp_dim, self.use_bias)
        )
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for attn, ff in self.layers:
            x = attn(x) + x  
            x = ff(x) + x    
        return self.norm(x)


class ViT(nn.Module):
    image_size: int 
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    groups: int
    mlp_dim: int
    channels: int = 3
    dim_head: int = 64

    def setup(self):
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        assert image_height % patch_height == 0, "image height must be divisible by patch height, but got image height:{} and patch height:{}".format(image_height, patch_height)
        assert image_width % patch_width == 0, "image width must be divisible by patch width, but got image width:{} and patch width:{}".format(image_width, patch_width)
        
        self.to_patch_embedding = nn.Sequential([
            lambda x: rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', 
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
            groups=self.groups,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim
        )
        self.cls_token = self.param('cls_token',
                                    nn.initializers.zeros,
                                    (1, 1, self.dim))
        
        self.linear_head = nn.Dense(self.num_classes)

    def __call__(self, img, train: bool = True):
        x = self.to_patch_embedding(img)
        
        x = jnp.concatenate([self.cls_token, x], axis=1)
        x += self.pos_embedding
        
        x = self.transformer(x)
        
        x = x[:, 0]
        
        return self.linear_head(x)