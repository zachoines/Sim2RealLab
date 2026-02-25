"""Tests for the strafer_description URDF model."""

import os
import xml.etree.ElementTree as ET

import pytest
import xacro

from strafer_shared.constants import (
    WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_BASE, TRACK_WIDTH,
    CHASSIS_LENGTH, CHASSIS_WIDTH, CHASSIS_HEIGHT, CHASSIS_GROUND_CLEARANCE,
    CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z,
    CAMERA_LENGTH, CAMERA_WIDTH, CAMERA_HEIGHT,
    MASS_WHEEL_ASSEMBLY, MASS_CHASSIS, MASS_CAMERA,
)


XACRO_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "urdf", "strafer.urdf.xacro"
)

# Mappings matching what description.launch.py passes
LAUNCH_MAPPINGS = {
    "wheel_radius": str(WHEEL_RADIUS),
    "wheel_width": str(WHEEL_WIDTH),
    "wheel_base": str(WHEEL_BASE),
    "track_width": str(TRACK_WIDTH),
    "chassis_length": str(CHASSIS_LENGTH),
    "chassis_width": str(CHASSIS_WIDTH),
    "chassis_height": str(CHASSIS_HEIGHT),
    "chassis_ground_clearance": str(CHASSIS_GROUND_CLEARANCE),
    "camera_x": str(CAMERA_OFFSET_X),
    "camera_y": str(CAMERA_OFFSET_Y),
    "camera_z": str(CAMERA_OFFSET_Z),
    "camera_length": str(CAMERA_LENGTH),
    "camera_width": str(CAMERA_WIDTH),
    "camera_height": str(CAMERA_HEIGHT),
    "mass_wheel_assembly": str(MASS_WHEEL_ASSEMBLY),
    "mass_chassis": str(MASS_CHASSIS),
    "mass_camera": str(MASS_CAMERA),
}


@pytest.fixture
def urdf_root():
    """Process xacro with constants mappings and return the XML root element."""
    urdf_xml = xacro.process_file(XACRO_PATH, mappings=LAUNCH_MAPPINGS).toxml()
    return ET.fromstring(urdf_xml)


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
    urdf_xml = xacro.process_file(XACRO_PATH, mappings=LAUNCH_MAPPINGS).toxml()
    assert len(urdf_xml) > 0


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


def test_camera_mount_position(urdf_root):
    """Camera should be at CAMERA_OFFSET_X/Y/Z from constants."""
    joint = urdf_root.find(".//joint[@name='d555_mount']")
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.attrib["xyz"].split()]
    assert xyz[0] == pytest.approx(CAMERA_OFFSET_X)
    assert xyz[1] == pytest.approx(CAMERA_OFFSET_Y)
    assert xyz[2] == pytest.approx(CAMERA_OFFSET_Z)


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
