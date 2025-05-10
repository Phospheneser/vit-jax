import copy
import random
from functools import wraps, partial

import jax
from jax import tree_util


import jax
from flax import linen as nn
from flax.core import freeze, unfreeze
from typing import Optional
import torch.nn.functional as F

from torchvision import transforms as T

# helper functions

def exists(val):
    return val is not None

def default(val, default):
    return val if exists(val) else default

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(params):
    leaves = tree_util.tree_leaves(params)
    if not leaves:
        raise ValueError("Module has no parameters.")
    return leaves[0].device()

def set_requires_grad(params, val):
    """通过stop_gradient控制梯度传播"""
    def apply_grad_control(p):
        return p if val else jax.lax.stop_gradient(p)
    return tree_util.tree_map(apply_grad_control, params)


# loss function # (algorithm 1 in the paper)

def loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
    ):
    # 阻止教师logits的梯度传播
    teacher_logits = jax.lax.stop_gradient(teacher_logits)
    
    # 计算学生概率分布（带温度缩放）
    student_probs = jax.nn.softmax(student_logits / student_temp, axis=-1)
    
    # 计算教师概率分布（带中心校正和温度缩放）
    teacher_probs = jax.nn.softmax(
        (teacher_logits - centers) / teacher_temp, 
        axis=-1
    )
    
    # 计算KL散度形式的损失
    loss = - (teacher_probs * jnp.log(student_probs + eps)).sum(axis=-1).mean()
    
    return loss

# augmentation utils

class RandomApply(nn.Module):
    fn: callable  # 需要应用的变换函数
    p: float      # 应用概率

    @nn.compact
    def __call__(self, x, rng_key=None):
        # 获取或生成 RNG key
        if rng_key is None:
            # 使用 Flax 的 make_rng 机制生成密钥
            rng_key = self.make_rng('random_apply')
        else:
            # 显式分割传入的密钥
            rng_key, _ = jax.random.split(rng_key)
        
        # 生成伯努利采样结果
        should_apply = jax.random.bernoulli(rng_key, self.p)
        
        # 条件应用
        return jax.lax.cond(
            should_apply,
            lambda: self.fn(x),
            lambda: x
        )

# exponential moving average

class EMA(nn.Module):
    beta: float
    
    def update_average(self, old: jax.Array | None, new: jax.Array) -> jax.Array:
        """JAX要求返回新值而非就地修改"""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    @nn.compact
    def __call__(self, ma_params: dict, current_params: dict) -> dict:
        """参数更新通过函数式操作实现"""
        return jax.tree_map(
            lambda old, new: self.update_average(old, new),
            ma_params, current_params
        )

@jax.jit
def update_moving_average(ema: EMA, ma_params: dict, current_params: dict) -> dict:
    """使用JAX树操作代替PyTorch的in-place更新"""
    # 停止当前模型参数的梯度传播
    current_params = jax.lax.stop_gradient(current_params)
    return ema.apply({'params': ma_params}, current_params)

class L2Norm(nn.Module):
    eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        norm = jnp.linalg.norm(x, axis=1, keepdims=True)
        norm = jnp.clip(norm, a_min=self.eps)
        return x / norm

class MLP(nn.Module):
    dim_out: int
    num_layers: int
    hidden_size: int = 256
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dim = x.shape[-1]
        layers = []
        
        # 动态构建隐藏层
        for i in range(self.num_layers - 1):
            layers.append(
                nn.Dense(self.hidden_size if i < self.num_layers-2 else self.dim_out)
            )
            if i < self.num_layers-2:
                layers.append(nn.gelu)
        
        # 最终层结构
        x = nn.Sequential(layers)(x)
        x = L2Norm()(x)
        return nn.Dense(self.dim_out)(x)  # 最后一层线性变换

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets


class NetWrapper(nn.Module):
    net: nn.Module
    layer: int = -2
    output_dim: int = 256
    projection_hidden_size: int = 256
    projection_num_layers: int = 2
    
    def setup(self):
        # 使用Flax的lazy初始化机制动态创建projector
        self._projector = None
        self._intermediate_output = None
        
        # 递归查找目标层
        def find_layer(module, path):
            if isinstance(path, str):
                return getattr(module, path, None)
            elif isinstance(path, int):
                children = [c for c in module.children() if isinstance(c, nn.Module)]
                return children[path] if path < len(children) else None
            return None
            
        self.target_layer = find_layer(self.net, self.layer)
        assert self.target_layer is not None, f"Layer {self.layer} not found"

    @nn.compact
    def __call__(self, x, return_projection: bool = True):
        # 重写前向传播流程
        def modified_forward(module, x):
            if module is self.target_layer:
                # 捕获中间层输出
                self._intermediate_output = module(x)
                return self._intermediate_output
            return module(x)

        # 运行网络并获取中间输出
        final_output = self._apply_with_interception(self.net, modified_forward, x)
        
        if return_projection:
            # 动态初始化projector
            if self._projector is None and self._intermediate_output is not None:
                input_dim = self._intermediate_output.shape[-1]
                self._projector = MLP(
                    dim=input_dim,
                    dim_out=self.output_dim,
                    num_layers=self.projection_num_layers,
                    hidden_size=self.projection_hidden_size
                )
            # 返回projection和原始embedding
            return self._projector(self._intermediate_output), final_output
        return final_output

    def _apply_with_interception(self, module, hook_fn, x):
        # 递归应用hook函数
        if module is self.target_layer:
            return hook_fn(module, x)
        
        if isinstance(module, nn.Sequential):
            return jax.lax.stop_gradient(
                nn.Sequential([self._apply_with_interception(m, hook_fn, x) for m in module.layers])
            )
        elif hasattr(module, 'children'):
            for child in module.children():
                return self._apply_with_interception(child, hook_fn, x)
        return module(x)



class DINO(nn.Module):
    image_size: int
    projection_hidden_size: int = 256
    num_classes_K: int = 65336  
    student_temp: float = 0.9
    teacher_temp: float = 0.04
    moving_average_decay: float = 0.9
    center_moving_average_decay: float = 0.9

    def setup(self):
        # 数据增强链
        self.augment1 = self._build_augmentation_chain()
        self.augment2 = self._build_augmentation_chain()

        # 网络结构
        self.student_encoder = NetWrapper(
            self._build_base_network(), 
            self.projection_hidden_size,
            projection_num_layers=4,
            output_dim=self.num_classes_K
        )
        
        # EMA初始化
        self.teacher_ema_updater = create_ema_updater(self.moving_average_decay)
        self.center_ema_updater = create_ema_updater(self.center_moving_average_decay)

    def _build_base_network(self) -> nn.Module:
        # 示例ViT结构，需替换为实际基础网络
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7,7)),
            nn.relu,
            nn.max_pool,
            nn.Dense(512),
            nn.LayerNorm()
        ])

    def _build_augmentation_chain(self):
        # 使用jax.image实现数据增强
        return nn.Sequential([
            partial(jax.image.resize, shape=(self.image_size, self.image_size, 3)),
            partial(random_color_jitter, strength=0.8),
            partial(random_blur, kernel_size=3)
        ])

    def _crop_transform(self, rng, image, scale_range):
        # 随机裁剪实现
        scale = jax.random.uniform(rng, minval=scale_range[0], maxval=scale_range[1])
        new_size = int(self.image_size * scale)
        return jax.image.resize(image, (new_size, new_size, 3))

    def __call__(self, state: DinoState, x):
        # 数据增强分支
        key1, key2 = jax.random.split(state['key'])
        
        # 局部和全局裁剪
        local_crop = partial(self._crop_transform, scale_range=(0.05, 0.4))
        global_crop = partial(self._crop_transform, scale_range=(0.5, 1.0))
        
        # 前向计算
        student_proj1 = self.student_encoder(local_crop(key1, x))
        student_proj2 = self.student_encoder(local_crop(key2, x))
        
        # 教师网络EMA
        teacher_proj1 = self.teacher_encoder(global_crop(key1, x))
        teacher_proj2 = self.teacher_encoder(global_crop(key2, x))
        
        # 损失计算
        loss = (self._dino_loss(teacher_proj1, student_proj2) + 
                self._dino_loss(teacher_proj2, student_proj1)) / 2
        
        # 状态更新
        new_teacher = self.teacher_ema_updater(state['teacher_params'], 
                                             self.student_encoder.params)
        new_centers = self.center_ema_updater(state['centers'], 
                                            teacher_proj1.mean(0))
        
        return loss, {
            'teacher_params': new_teacher,
            'centers': new_centers,
            'key': key2
        }

    def _dino_loss(self, teacher, student):
        # 中心化处理
        teacher = teacher - self.centers
        teacher = teacher / self.teacher_temp
        student = student / self.student_temp
        
        return optax.softmax_cross_entropy(
            jax.nn.softmax(teacher, axis=-1),
            jax.nn.log_softmax(student, axis=-1)
        ).mean()