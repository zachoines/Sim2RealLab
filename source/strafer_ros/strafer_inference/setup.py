from setuptools import setup
import os
from glob import glob

package_name = "strafer_inference"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
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
    description=(
        "Trained-policy execution backend for the Strafer robot. Loads an "
        "exported policy and drives /strafer/cmd_vel from a /navigate_to_pose "
        "goal, providing the strafer_direct alternative to the Nav2 backend."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = strafer_inference.inference_node:main",
        ],
    },
)
