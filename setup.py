from setuptools import find_packages, setup

setup(
    name="dornet",
    version="2026.03.04",
    author="yanzq95",
    description="DORNet: A Degradation Oriented and Regularized Network for Blind Depth Super-Resolution (CVPR 2025)",
    long_description=open("dornet/README.md").read(),
    long_description_content_type="text/markdown",
    license="file: dornet/LICENSE",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy<2.0",
        "torchvision>=0.16.0",
        "scipy>=1.11.3",
        "Pillow>=10.0.1",
        "tqdm>=4.65.0",
        "scikit-image>=0.21.0",
        "mmcv-full==1.7.2",
        "openmim>=0.3.9",
    ],
    dependency_links=["https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html"],
    include_package_data=True,
    package_data={
        "dornet": ["checkpoints/*.pth"],
    },
)
