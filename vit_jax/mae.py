import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import repeat

from vit_jax.vit import Transformer

class MAE(nn.Module):
    encoder: nn.Module  # JAX风格的模块组合
    decoder_dim: int
    masking_ratio: float = 0.75
    decoder_depth: int = 1
    decoder_heads: int = 8
    decoder_dim_head: int = 64

    def setup(self):
        # 初始化与PyTorch对应的参数
        assert 0 < self.masking_ratio < 1, "Masking ratio must be between 0 and 1"
        
        # 从encoder提取必要参数
        num_patches = self.encoder.pos_embedding.shape[1]
        encoder_dim = self.encoder.pos_embedding.shape[2]
        pixel_dim = self.encoder.patch_embed.proj.kernel.shape[-1]  # 假设使用Conv投影
        
        # 解码器组件
        self.enc_to_dec = nn.Dense(self.decoder_dim) if encoder_dim != self.decoder_dim else lambda x: x
        self.mask_token = self.param('mask_token', nn.initializers.normal(stddev=0.02), (self.decoder_dim,))
        self.decoder = Transformer(self.decoder_dim, self.decoder_depth, self.decoder_heads, self.decoder_dim_head)
        self.decoder_pos_emb = nn.Embed(num_patches, self.decoder_dim)
        self.to_pixels = nn.Dense(pixel_dim)

    def __call__(self, images, deterministic=True, rng=None):
        # 图像分块处理
        patches = self.encoder.patch_embed(images)
        batch, num_patches, _ = patches.shape
        
        # 添加位置编码
        pos_emb = self.encoder.pos_embedding[:, 1:]  # 排除CLS token
        tokens = patches + pos_emb

        # 生成随机掩码
        rng, mask_rng = jax.random.split(rng)
        rand = jax.random.uniform(mask_rng, (batch, num_patches))
        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = jnp.argsort(rand, axis=-1)[:, :num_masked]
        unmasked_indices = jnp.argsort(rand, axis=-1)[:, num_masked:]

        # 编码可见块
        batch_indices = jnp.arange(batch)[:, None]
        unmasked_tokens = tokens[batch_indices, unmasked_indices]
        encoded = self.encoder.transformer(unmasked_tokens, deterministic=deterministic)

        # 解码器准备
        decoder_tokens = self.enc_to_dec(encoded)
        decoder_tokens += self.decoder_pos_emb(unmasked_indices)

        # 构建完整解码器输入
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens += self.decoder_pos_emb(masked_indices)
        
        # 合并可见和掩码token
        all_tokens = jnp.zeros((batch, num_patches, self.decoder_dim))
        all_tokens = all_tokens.at[batch_indices, unmasked_indices].set(decoder_tokens)
        all_tokens = all_tokens.at[batch_indices, masked_indices].set(mask_tokens)
        
        # 解码过程
        decoded = self.decoder(all_tokens)
        pred_pixels = self.to_pixels(decoded[batch_indices, masked_indices])

        # 计算重建损失
        target_pixels = patches[batch_indices, masked_indices]
        loss = jnp.mean((pred_pixels - target_pixels) ** 2)
        return loss
