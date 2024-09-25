from setuptools import setup, find_packages

setup(
    name="scar",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.3',
        'transformers>=4.37',
        'huggingface_hub>=0.23',
        'scikit-learn',
        'tqdm',
        'nltk',
        'datasketch'
    ],
    author="Anonymous",
    author_email="anon@example.com",
    description="SCAR: An AI-powered tool for ranking and filtering instruction-answer pairs based on writing quality and style consistency",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anon/scar",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
