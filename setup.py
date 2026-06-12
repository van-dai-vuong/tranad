from setuptools import setup

setup(
    name="tranad",
    version="0.1.0",
    description="TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series (packaged as a library)",
    # Map the repo root to the 'tranad' package so you can `import tranad`
    package_dir={"tranad": "."},
    packages=["tranad", "tranad.src"],
    package_data={"tranad.src": ["params.json"]},
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
    ],
    python_requires=">=3.8",
)
