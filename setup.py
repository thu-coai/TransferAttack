from setuptools import setup, find_packages

setup(
    name="TransferAttack",
    version="0.0.dev",
    description="Transferable attack",
    author="Junxiao Yang, Zhexin Zhang, et al.",
    author_email="yangjunx21@gmail.com",
    url="https://github.com/thu-coai/TransferAttack",
    packages=find_packages(include=('llm_attacks*',)),
    include_package_data=True,
    install_requires=[
        'transformers>=4.34.0',
        'torch>=2.0',
        'openai>=1.0.0',
        'numpy',
        'fschat',
        'tokenizers >= 0.13.3',
    ],
    python_requires=">=3.7",
    keywords=['ai safety', 
             ],
    license='MIT',
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3"
    ]
)