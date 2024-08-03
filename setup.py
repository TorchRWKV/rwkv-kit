from setuptools import setup, find_packages

setup(
    name="torchrwkv",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'torchrwkv': ['assets/*'],
    },
    install_requires=[
        "torch>=2.0.0,<3.0.0",
        "fla @ git+https://gitee.com/uniartisan2018/flash-linear-attention.git",
    ],
    author="Yang Xiao, Zhiyuan Li",
    author_email="jiaxinsugar@gmail.com, uniartisan2017@gmail.com",
    description="TorchRWKV is a pure PyTorch implementation of the RWKV.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TorchRWKV/torchrwkv",  # 更新为正确的 GitHub 仓库 URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)