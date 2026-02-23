from setuptools import setup, find_packages
import os
from glob import glob

package_name = "strafer_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="jetson",
    maintainer_email="jetson@localhost.local",
    description="RealSense D555 launch wrapper and depth downsampler for the Strafer robot",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "depth_downsampler = strafer_perception.depth_downsampler:main",
        ],
    },
)
