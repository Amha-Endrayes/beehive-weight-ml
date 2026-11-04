from setuptools import setup, find_packages

setup(
    name="beehive_weight_ml",
    version="1.0.0",
    author="Amha Endrayes, Natnael Behailu",
    author_email="research@robox-ai.org",
    description="Machine Learning Investigation into Beehive Weight Estimation Without a Scale",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Robox-AI/beehive-weight-ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "joblib>=1.2.0",
    ],
)
