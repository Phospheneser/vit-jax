import jax
import jax.numpy as jnp
from vit_jax import ViT  

def test():
    rng = jax.random.PRNGKey(0)
    init_rng, data_rng = jax.random.split(rng)
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        groups = 2,
        mlp_dim = 2048
    )
    variables = v.init(init_rng, jnp.ones([1, 256, 256, 3]))  
    img = jax.random.normal(data_rng, (1, 256, 256, 3))  
    @jax.jit
    def forward_fn(params, inputs):
        return v.apply({'params': params}, inputs, train=False) 
    
    preds = forward_fn(variables['params'], img)
    assert preds.shape == (1, 1000), 'Correct logits output shape'

if __name__ == "__main__":
    test()
    print("Test passed!")