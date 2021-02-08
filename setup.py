from distutils.core import setup

setup(
    name='thesis-tools',
    version='0.8.0',
    install_requires=[
        'pandas',
        'pytorch-nlp',
        'pytorch_lightning',
        'sphinx',
        'sphinx-rtd-theme',
        'tokenizers',
        'transformers'
    ]
)
