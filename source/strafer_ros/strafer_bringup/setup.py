from setuptools import setup
import os
from glob import glob

package_name = "strafer_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.env")),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="jetson",
    maintainer_email="jetson@localhost.local",
    description="Composed launch files for the GoBilda Strafer robot",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "validate_drive = strafer_bringup.validate_drive:main",
        ],
    },
)
