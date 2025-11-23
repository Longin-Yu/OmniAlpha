# setup.py
from pathlib import Path
from setuptools import setup, find_packages

# here = Path(__file__).parent
# readme = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name="alpha",
    version="0.1.0",
    description="alpha package",
    # long_description=readme,
    # long_description_content_type="text/markdown",
    packages=find_packages(include=["alpha", "alpha.*"]),  # 只打包 alpha 目录
    # include_package_data=True,  # 如需携带非 .py 文件，可配合 package_data
    # package_data={"alpha": ["**/*.json", "**/*.yaml", "**/*.yml", "**/*.txt"]},
    install_requires=[],         # 依赖写这里
    python_requires=">=3.8",
)
