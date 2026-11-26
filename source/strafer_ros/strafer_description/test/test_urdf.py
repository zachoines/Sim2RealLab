"""Tests for the strafer_description URDF model."""

import importlib.util
import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import pytest
import xacro

from strafer_description.xacro_args import robot_description, xacro_mappings
from strafer_shared.constants import (
    WHEEL_BASE, TRACK_WIDTH,
    CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z,
    MASS_WHEEL_ASSEMBLY, MASS_CHASSIS, MASS_CAMERA,
    FRAME_BASE_LINK, FRAME_D555_LINK, FRAME_D555_COLOR_OPTICAL,
)


PACKAGE_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
XACRO_PATH = os.path.join(PACKAGE_DIR, "urdf", "strafer.urdf.xacro")
LAUNCH_FILE = os.path.join(PACKAGE_DIR, "launch", "description.launch.py")


@pytest.fixture
def urdf_root():
    """Process the xacro as the launch file does and return the XML root element."""
    return ET.fromstring(robot_description(XACRO_PATH))


def _load_launch_file():
    """Import description.launch.py from the source tree."""
    spec = importlib.util.spec_from_file_location("description_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _origin(urdf_root, joint_name):
    """Return a joint's origin xyz and rpy as floats, zero where the URDF leaves one out."""
    origin = urdf_root.find(f".//joint[@name='{joint_name}']/origin")
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    return xyz, rpy


def test_xacro_processes_standalone():
    """Xacro file should process with sentinel defaults (no mappings)."""
    urdf_xml = xacro.process_file(XACRO_PATH).toxml()
    assert len(urdf_xml) > 0


def test_sentinel_defaults_without_mappings():
    """Without mappings, sentinel values (-1) should appear in the URDF."""
    urdf_xml = xacro.process_file(XACRO_PATH).toxml()
    root = ET.fromstring(urdf_xml)
    chassis = root.find(".//link[@name='chassis_link']/inertial/mass")
    assert float(chassis.attrib["value"]) == pytest.approx(-1.0)


def test_xacro_processes_with_mappings():
    """Xacro file should process with constants mappings (launch path)."""
    assert len(robot_description(XACRO_PATH)) > 0


def test_mappings_cover_every_xacro_arg():
    """Every xacro:arg has a mapping and every mapping an arg, so no -1 default reaches the URDF."""
    args = {
        arg.getAttribute("name")
        for arg in minidom.parse(XACRO_PATH).getElementsByTagName("xacro:arg")
    }
    assert set(xacro_mappings()) == args


def test_launch_file_uses_robot_description(monkeypatch):
    """robot_state_publisher receives robot_description() of the package's xacro."""
    description_launch = _load_launch_file()
    assert description_launch.robot_description is robot_description

    xacro_paths = []
    nodes = []
    real_description = description_launch.robot_description
    real_node = description_launch.Node

    def share_directory(package):
        assert package == "strafer_description"
        return PACKAGE_DIR

    def spy_description(xacro_path):
        xacro_paths.append(xacro_path)
        return real_description(xacro_path)

    def record_node(**kwargs):
        nodes.append(kwargs)
        return real_node(**kwargs)

    monkeypatch.setattr(description_launch, "get_package_share_directory", share_directory)
    monkeypatch.setattr(description_launch, "robot_description", spy_description)
    monkeypatch.setattr(description_launch, "Node", record_node)
    description_launch.generate_launch_description()

    assert len(xacro_paths) == 1
    assert os.path.samefile(xacro_paths[0], XACRO_PATH)
    rsp = [kwargs for kwargs in nodes if kwargs["package"] == "robot_state_publisher"]
    assert len(rsp) == 1
    assert len(rsp[0]["parameters"]) == 1
    assert rsp[0]["parameters"][0]["robot_description"] == robot_description(XACRO_PATH)


def test_robot_name(urdf_root):
    assert urdf_root.attrib["name"] == "strafer"


def test_expected_links(urdf_root):
    link_names = {link.attrib["name"] for link in urdf_root.findall("link")}
    expected = {
        "base_link",
        "chassis_link",
        "wheel_1_link",
        "wheel_2_link",
        "wheel_3_link",
        "wheel_4_link",
        "d555_link",
        "d555_color_optical_frame",
    }
    assert expected == link_names


def test_expected_joints(urdf_root):
    joints = {j.attrib["name"]: j.attrib["type"] for j in urdf_root.findall("joint")}
    assert joints["chassis_joint"] == "fixed"
    assert joints["wheel_1_drive"] == "continuous"
    assert joints["wheel_2_drive"] == "continuous"
    assert joints["wheel_3_drive"] == "continuous"
    assert joints["wheel_4_drive"] == "continuous"
    assert joints["d555_mount"] == "fixed"


def test_wheel_joint_axes(urdf_root):
    """All wheel joints should rotate around the Y axis."""
    for name in ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]:
        joint = urdf_root.find(f".//joint[@name='{name}']")
        axis = joint.find("axis")
        assert axis.attrib["xyz"] == "0 1 0", f"{name} axis should be Y"


def test_wheel_positions_symmetric(urdf_root):
    """Wheels should be placed symmetrically around base_link."""
    positions = {}
    for name in ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]:
        joint = urdf_root.find(f".//joint[@name='{name}']")
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.attrib["xyz"].split()]
        positions[name] = xyz

    # FL and FR at same X (front), RL and RR at same X (rear)
    assert positions["wheel_1_drive"][0] == pytest.approx(positions["wheel_2_drive"][0])
    assert positions["wheel_3_drive"][0] == pytest.approx(positions["wheel_4_drive"][0])

    # FL and RL at same Y (left), FR and RR at same Y (right)
    assert positions["wheel_1_drive"][1] == pytest.approx(positions["wheel_3_drive"][1])
    assert positions["wheel_2_drive"][1] == pytest.approx(positions["wheel_4_drive"][1])

    # Left and right are mirror images
    assert positions["wheel_1_drive"][1] == pytest.approx(
        -positions["wheel_2_drive"][1]
    )

    # Front-rear distance matches WHEEL_BASE from constants
    dx = positions["wheel_1_drive"][0] - positions["wheel_3_drive"][0]
    assert dx == pytest.approx(WHEEL_BASE, abs=0.001)

    # Left-right distance matches TRACK_WIDTH from constants
    dy = positions["wheel_1_drive"][1] - positions["wheel_2_drive"][1]
    assert dy == pytest.approx(TRACK_WIDTH, abs=0.001)


def test_camera_mount_joint(urdf_root):
    """d555_mount fixes d555_link to base_link."""
    joint = urdf_root.find(".//joint[@name='d555_mount']")
    assert joint.attrib["type"] == "fixed"
    assert joint.find("parent").attrib["link"] == FRAME_BASE_LINK
    assert joint.find("child").attrib["link"] == FRAME_D555_LINK


def test_camera_mount_position(urdf_root):
    """d555_mount's origin is exactly CAMERA_OFFSET_X/Y/Z, with no rotation.

    The SLAM database key reads this origin from the same three constants, so a
    rotation or another input here needs the key extended in the same change.
    """
    xyz, rpy = _origin(urdf_root, "d555_mount")
    assert xyz == [CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z]
    assert rpy == [0.0, 0.0, 0.0]


def test_color_optical_joint(urdf_root):
    """d555_color_optical_frame sits on d555_link, turned to x right, y down, z forward.

    The SLAM database key does not cover this joint, so an offset or rotation here
    needs the key extended in the same change.
    """
    joint = urdf_root.find(".//joint[@name='d555_color_optical_joint']")
    assert joint.attrib["type"] == "fixed"
    assert joint.find("parent").attrib["link"] == FRAME_D555_LINK
    assert joint.find("child").attrib["link"] == FRAME_D555_COLOR_OPTICAL
    xyz, rpy = _origin(urdf_root, "d555_color_optical_joint")
    assert xyz == pytest.approx([0.0, 0.0, 0.0])
    assert rpy == pytest.approx([-math.pi / 2, 0.0, -math.pi / 2])


def test_masses_from_constants(urdf_root):
    """Link masses should match constants.py values."""
    def _mass(link_name):
        link = urdf_root.find(f".//link[@name='{link_name}']")
        return float(link.find("inertial/mass").attrib["value"])

    assert _mass("chassis_link") == pytest.approx(MASS_CHASSIS)
    assert _mass("d555_link") == pytest.approx(MASS_CAMERA)
    for i in range(1, 5):
        assert _mass(f"wheel_{i}_link") == pytest.approx(MASS_WHEEL_ASSEMBLY)


def test_no_disconnected_links(urdf_root):
    """Every non-root link should be a child of exactly one joint."""
    child_links = {
        j.find("child").attrib["link"] for j in urdf_root.findall("joint")
    }
    all_links = {link.attrib["name"] for link in urdf_root.findall("link")}
    # base_link is root (not a child of any joint)
    orphaned = all_links - child_links - {"base_link"}
    assert orphaned == set(), f"Orphaned links: {orphaned}"
