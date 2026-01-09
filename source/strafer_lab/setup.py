"""Setup script for strafer_lab extension."""

import toml
from setuptools import setup

# Read pyproject.toml for metadata
with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

project = pyproject["project"]

setup(
    name=project["name"],
    version=project["version"],
    description=project["description"],
    author=project["authors"][0]["name"],
    author_email=project["authors"][0]["email"],
    packages=["strafer_lab", "strafer_lab.assets", "strafer_lab.tasks", "strafer_lab.tasks.navigation"],
    package_dir={"": "."},
    python_requires=project["requires-python"],
    install_requires=project["dependencies"],
)
