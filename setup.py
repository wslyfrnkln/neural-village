from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="neural-village",
    version="0.1.0",
    description="Run your music through a brain. Predict neural activation patterns from audio using Meta FAIR's TRIBE v2.",
    author="WESLEYFRANKLIN",
    url="https://github.com/wslyfrnkln/neural-village",
    packages=find_packages(exclude=["tests*", "cli*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "neural-village=cli.main:analyze",
        ],
    },
)
