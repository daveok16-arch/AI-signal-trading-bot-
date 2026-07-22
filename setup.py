"""Setup script for XAUUSD Scalping System."""
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="xauusd-scalper",
    version="1.0.0",
    author="XAUUSD AI Scalper",
    author_email="",
    description="AI-Powered XAUUSD Scalping Signal System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xauusd-scalper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "PyYAML>=6.0",
        "yfinance>=0.2.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "streamlit-autorefresh>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "python-dateutil>=2.8.0",
        "numba>=0.57.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "mlfinlab": ["mlfinlab>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "xauusd-train=scripts.train:main",
            "xauusd-backtest=scripts.backtest:main",
            "xauusd-api=src.api.main:main",
            "xauusd-dashboard=src.dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml"],
    },
)
