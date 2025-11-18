from setuptools import setup, find_packages

setup(
    name="adaptive-agent",
    version="1.0.0",
    description="A fully functional personalized AI agent that learns and adapts to users",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "adaptive-agent=src.main:main",
        ],
    },
)
