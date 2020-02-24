from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name = 'hyanova',
    version = '1.0.0',
    keywords = ('anova', 'sklearn','hyperparameter','hyperparameter importance'),
    description = 'A pure python implementation of fuctional ANOVA algorithm.',
    license = 'MIT',
    install_requires = ['numpy', 'pandas', 'tqdm'],
    author = 'Su Qiao',
    author_email = 'qiaosu98@outlook.com',
    url = 'https://github.com/exiarepairii/hyanova',
    packages = find_packages(),
    platforms = 'any',
)
