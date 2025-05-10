
from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
  name = 'vit-jax',
  packages = find_packages(exclude=['examples']),
  version = '0.0.0',
  license='MIT',
  description = 'Vision Transformer (ViT) - Jax',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = 'Phospheneser',
  author_email = 'zxu24@m.fudan.edu.cn',
  url = 'https://github.com/Phospheneser/vit-jax',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'image recognition',
    'jax'
  ],
  install_requires=[
    'einops>=0.7.0',
    'jax>=0.6.0',
    'jaxlib>=0.6.0'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest',
    'jax==0.6.0',
    'jaxlib==0.6.0',
    'einops==0.8.1',
    'flax==0.10.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.12',
  ],
)
