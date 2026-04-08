from setuptools import setup
import os

# Helper to read README
def read_file(filename):
    if os.path.exists(filename):
        with open(filename, encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="dornet",
    version="2026.04.08",
    author="yanzq95",
    description="DORNet: A Degradation Oriented and Regularized Network for Blind Depth Super-Resolution (CVPR 2025)",
    long_description=read_file("dornet/README.md"),
    long_description_content_type="text/markdown",
    license=read_file("dornet/LICENSE"),
    packages=['dornet', 'dornet.net'],
    python_requires=">=3.10",

    install_requires=[
        "torch>=2.1.0",
        "numpy<2.0",
        "torchvision>=0.16.0",
        "scipy>=1.11.3",
        "Pillow>=10.0.1",
        "tqdm>=4.65.0",
        "scikit-image>=0.21.0",
        # "mmcv-full",
    ],

    include_package_data=True,
    package_data={
        "dornet": ["checkpoints/*.pth"],
    },
)
