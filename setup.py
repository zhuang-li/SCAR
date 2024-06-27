from setuptools import setup, find_packages

setup(
    name="scar-tool",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'scikit-learn',
        'tqdm',
        'nltk',
        'datasketch',
        'peft',
        'trl',
        'accelerate',
        'py-cpuinfo'
        'deepspeed'
    ],
    author="Zhuang Li",
    author_email="zhuang.li1@monash.edu",
    description="SCAR: An AI-powered tool for ranking and filtering instruction-answer pairs based on writing quality and style consistency",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhuang-li/SCAR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)