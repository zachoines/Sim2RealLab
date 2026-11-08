"""Pins the autonomy container's Cyclone DDS file: the tracked receive tuning plus a 3 s
participant lease, mounted for `autonomy` alone at the container path every service reads.
"""

import os
import re
import xml.etree.ElementTree as ET

import pytest
import yaml

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.normpath(os.path.join(TEST_DIR, "..", "config"))
DEPLOY_DIR = os.path.normpath(os.path.join(TEST_DIR, "..", "..", "deploy"))
TRACKED = os.path.join(CONFIG_DIR, "cyclonedds.xml")
AUTONOMY = os.path.join(CONFIG_DIR, "cyclonedds.autonomy.xml")
BASE_COMPOSE = "docker-compose.yml"
COMPOSE_FILES = sorted(
    f for f in os.listdir(DEPLOY_DIR)
    if f.startswith("docker-compose") and f.endswith(".yml") and f != "docker-compose.override.yml"
)

NS = "{https://cdds.io/config}"
LEASE = "3 s"
_SHORT_VOLUME = re.compile(r"^(?P<source>.+?):(?P<target>/[^:]*)(?::(?P<mode>[\w,]+))?$")
_CYCLONE_FILE = re.compile(r"^cyclonedds(\.\w+)?\.xml$")


def _shape(elem):
    return (elem.tag, sorted(elem.attrib.items()), (elem.text or "").strip(),
            [_shape(child) for child in elem])


def _compose(filename):
    with open(os.path.join(DEPLOY_DIR, filename)) as f:
        return yaml.safe_load(f) or {}


def _container_path():
    uri = _compose(BASE_COMPOSE)["x-dds-env"]["CYCLONEDDS_URI"]
    assert uri.startswith("file://")
    return uri[len("file://"):]


def _mounts(svc):
    for entry in (svc or {}).get("volumes") or []:
        if isinstance(entry, str):
            m = _SHORT_VOLUME.match(entry)
            if not m:
                continue
            source, target = m.group("source"), m.group("target")
            read_only = "ro" in (m.group("mode") or "").split(",")
        else:
            source, target = str(entry.get("source", "")), str(entry.get("target", ""))
            read_only = bool(entry.get("read_only"))
        if source.startswith((".", "/")):
            source = os.path.normpath(os.path.join(DEPLOY_DIR, source))
        yield source, target, read_only


def _cyclone_mounts(svc):
    path = _container_path()
    return [(source, target, ro) for source, target, ro in _mounts(svc)
            if target == path or _CYCLONE_FILE.match(os.path.basename(source))]


class TestCycloneLeaseFile:

    def test_autonomy_file_parses(self):
        assert ET.parse(AUTONOMY).getroot().tag == f"{NS}CycloneDDS"

    def test_autonomy_file_sets_one_lease_of_3_s(self):
        root = ET.parse(AUTONOMY).getroot()
        leases = root.findall(f".//{NS}LeaseDuration")
        assert len(leases) == 1
        assert root.find(f"{NS}Domain/{NS}Discovery/{NS}LeaseDuration") is leases[0]
        assert leases[0].text.strip() == LEASE

    def test_autonomy_file_is_the_tracked_file_plus_the_lease(self):
        tracked, autonomy = ET.parse(TRACKED).getroot(), ET.parse(AUTONOMY).getroot()
        domain = autonomy.find(f"{NS}Domain")
        discovery = domain.find(f"{NS}Discovery")
        discovery.remove(discovery.find(f"{NS}LeaseDuration"))
        if (not len(discovery) and not discovery.attrib and not (discovery.text or "").strip()
                and tracked.find(f"{NS}Domain/{NS}Discovery") is None):
            domain.remove(discovery)
        assert _shape(autonomy) == _shape(tracked)

    def test_tracked_file_sets_no_lease(self):
        assert ET.parse(TRACKED).getroot().findall(f".//{NS}LeaseDuration") == []


class TestCycloneLeaseMount:

    def test_compose_files_found(self):
        assert {BASE_COMPOSE, "docker-compose.sim.yml"} <= set(COMPOSE_FILES)

    def test_autonomy_mounts_the_lease_file_at_the_uri_path(self):
        autonomy = _compose(BASE_COMPOSE)["services"]["autonomy"]
        assert _cyclone_mounts(autonomy) == [(AUTONOMY, _container_path(), True)]

    @pytest.mark.parametrize("filename", [f for f in COMPOSE_FILES if f != BASE_COMPOSE])
    def test_no_other_compose_file_replaces_the_autonomy_mount(self, filename):
        autonomy = (_compose(filename).get("services") or {}).get("autonomy")
        assert {source for source, _, _ in _cyclone_mounts(autonomy)} <= {AUTONOMY}

    @pytest.mark.parametrize("filename", COMPOSE_FILES)
    def test_every_other_service_mounts_the_tracked_file(self, filename):
        wrong = {
            name: mounts
            for name, svc in (_compose(filename).get("services") or {}).items()
            if name != "autonomy"
            for mounts in [_cyclone_mounts(svc)]
            if any((source, target) != (TRACKED, _container_path()) for source, target, _ in mounts)
        }
        assert not wrong, f"{filename}: Cyclone mounts other than the tracked file: {wrong}"
