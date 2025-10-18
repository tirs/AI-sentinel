from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-sentinel",
    version="1.0.0",
    author="AI Sentinel Team",
    description="Multimodal Explainable System for Detecting Digital Human Rights Violations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-sentinel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ai-sentinel-api=src.api.server:main",
            "ai-sentinel-train-nlp=scripts.train_nlp_model:main",
            "ai-sentinel-train-vision=scripts.train_vision_model:main",
        ],
    },
)