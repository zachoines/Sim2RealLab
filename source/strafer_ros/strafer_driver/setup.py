from setuptools import setup, find_packages
import os
from glob import glob

package_name = "strafer_driver"

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
        "pyserial",
    ],
    zip_safe=True,
    maintainer="jetson",
    maintainer_email="jetson@jetson-desktop",
    description="RoboClaw motor controller driver for the GoBilda Strafer chassis",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "roboclaw_node = strafer_driver.roboclaw_node:main",
        ],
    },
)
