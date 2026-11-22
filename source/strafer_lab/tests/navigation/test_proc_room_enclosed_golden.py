# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen-output guard for the enclosed room, which every depth stack trains on: the generator with its arguments.

Pins absolute values: one digest per case of the generator's output and its RNG stream position, driven with the
arguments of the enclosed room's ``generate_room`` term. The frozen manager hashes see those arguments but not the
code that consumes them, and the vanilla guard drives the generator at its defaults, where the enclosure, the
ceiling, the wall-biased clutter, the robot spawn pool and the placement phases never run.

A diff is a change to the enclosed room, not a test to update.

No Kit, no GPU. Run with the pure-Python lab tests.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch

from strafer_lab.tasks.navigation import strafer_env_cfg as _env_cfg
from strafer_lab.tasks.navigation.mdp import proc_room as _proc_room


class _CaptureEntity:
    def __init__(self):
        self.body_poses = None
        self.root_pose = None

    def write_body_link_pose_to_sim_index(self, body_poses, env_ids, body_ids):
        self.body_poses = body_poses.clone()

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.root_pose = root_pose.clone()


class _StubScene:
    def __init__(self, entities, env_origins):
        self._entities = entities
        self.env_origins = env_origins

    def __getitem__(self, key):
        return self._entities[key]


def _make_env(num_envs, difficulty):
    entities = {"room_primitives": _CaptureEntity(), "ceiling": _CaptureEntity()}
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=_StubScene(entities, torch.zeros(num_envs, 3)),
    )
    env._proc_room_difficulty = torch.full((num_envs,), difficulty, dtype=torch.long)
    return env, entities


def _enclosed_room_params():
    return dict(_env_cfg._GENERATE_PROC_ROOM_ENCLOSED.params)


SEEDS = (20260101, 20268020, 20275939, 20283858, 20291777, 20299696,
         20307615, 20315534, 20323453, 20331372, 20339291, 20347210)
# The enclosed room's events draw the difficulty from this range at each reset.
DIFFICULTIES = (4, 5, 6, 7)

# Every scalar draw is issued inside the per-env loop, so the stream is positionally coupled across envs and the
# batch size is swept; batch 64 spot-checks the range's ends.
CASES = tuple(
    (seed, difficulty, num_envs)
    for num_envs in (1, 8)
    for seed in SEEDS
    for difficulty in DIFFICULTIES
) + tuple(
    (seed, difficulty, 64)
    for seed in SEEDS[:3]
    for difficulty in (4, 7)
)


def _feed(h, tensor):
    t = tensor.detach().cpu().contiguous()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())


def _case_digest(module, params, seed, difficulty, num_envs):
    """Digest one call's observable output plus its stream position (the trailing draw)."""
    torch.manual_seed(seed)
    env, entities = _make_env(num_envs, difficulty)
    module.generate_proc_room(env, torch.arange(num_envs), **params)
    tail = torch.rand(4)

    h = hashlib.sha256()
    _feed(h, entities["room_primitives"].body_poses)
    _feed(h, entities["ceiling"].root_pose)
    _feed(h, env._proc_room_active_mask)
    _feed(h, env._proc_room_free_space)
    _feed(h, env._proc_room_spawn_pts)
    _feed(h, env._proc_room_spawn_count)
    _feed(h, env._proc_room_robot_spawn_pts)
    _feed(h, env._proc_room_robot_spawn_count)
    _feed(h, tail)
    return h.hexdigest()


def sweep_digests(module=_proc_room, params=None):
    """{case-key: digest} over the sweep, with the enclosed room's arguments unless ``params`` is given."""
    params = _enclosed_room_params() if params is None else params
    return {
        f"s{seed}-d{difficulty}-b{num_envs}": _case_digest(module, params, seed, difficulty, num_envs)
        for seed, difficulty, num_envs in CASES
    }


_SWEEP_GOLDEN = {
    "s20260101-d4-b1": "b31bb67a4551624fd926abb0d2b7cb2efea474ab18dcd4b2d70c35bb554b399d",
    "s20260101-d5-b1": "95d20293d0496727568cc77af7b5a16d7d1330e2815f415c3926b36f3b43d4ec",
    "s20260101-d6-b1": "e2d31a850d2d05300696438d05d9b2cae626c237098d89e23e9425e414719bd9",
    "s20260101-d7-b1": "420336aa4ca9abe1563f8822d798691ea1bdeaed0d3d4e595d37440b6ea22f11",
    "s20268020-d4-b1": "247b0e8ae87986146c3efe6fcc26d76012dea77a675e212ebf80e445257db7f4",
    "s20268020-d5-b1": "978e3da03f3416e05738767df3ff4557e8840fe26c66166b5e4e2b250014f931",
    "s20268020-d6-b1": "467daecc8576a806c3ec9d8dbde6746e888a02cd8c0810ad3da4053db8f1d13e",
    "s20268020-d7-b1": "90ab7b047b599173514a6a7e532452d6b70316e4a03c4d2dda17ed161f428aa9",
    "s20275939-d4-b1": "06bb5070426cfc026267c1280f6645a05fb506fa2e1394e4c473541f54f96290",
    "s20275939-d5-b1": "2ecc8fdda589933f0491bc5d6611e16d39a3737d73a6750fcf758b7c953b419f",
    "s20275939-d6-b1": "bc167d6dcef2ebec17350541691196f04e518c0d89feb26e73af20a6d1a7fc97",
    "s20275939-d7-b1": "a456f4ed95a4c1cf098a2315fab844d8e3b5af259bb6319e159e6cb795294178",
    "s20283858-d4-b1": "335a4b57494b53a82f7f465207fe3e3adb88cfc05c3d0734e24528fff929cfb8",
    "s20283858-d5-b1": "d27b600bf928568cfb46be9fba8de4be15ad88fe8c01de8af902e974ef761a3c",
    "s20283858-d6-b1": "800b48d58a42ca97352ae0a81debc6ec84d67df90211356c46a8a5812d694a8d",
    "s20283858-d7-b1": "04335ac076ae8e435726eeea222bd993b01480f10aaf9a7435e33f2756c82c34",
    "s20291777-d4-b1": "4f363c6b26e7f2603d016dbb10dc70a86f78318912003f0d672fd93240ccfa5f",
    "s20291777-d5-b1": "76990b33460549899a82edb4c9d368c6c8cacaccb0d45e4ca0cb7147a9703ab1",
    "s20291777-d6-b1": "406901f89170d6ae6605289dab95f9349f4f81021af70e3369e5759f6c5f72ca",
    "s20291777-d7-b1": "1dbf39bafc362705adae6389d7dce07871d8853372028302e08c38861aa5304d",
    "s20299696-d4-b1": "2f8a17ccacc2dc6941bf5a0384ffbe1cc4861b409ab199a384b6076586cbd326",
    "s20299696-d5-b1": "6b8df40f007a3b7e700a13311113d0d6904ac507b6e6cbb0cbc18c2e2b29a134",
    "s20299696-d6-b1": "5c23a05797cae5c3acd80994089e6a80bffd1e70208df4c00bcc694e40cd636d",
    "s20299696-d7-b1": "c0c8192081c7b33e02d98db0bc49f87688c4068f2efb508e9de32c2ff55205d0",
    "s20307615-d4-b1": "2ab80e12a5c5f2102af4ac7d97812df87a806cda8d7d8cd7b8a144e0265fedb6",
    "s20307615-d5-b1": "e33bd7715a5ee85c007744b9d33c8efaad32c769ba79efc5d4ee70f40e8b962f",
    "s20307615-d6-b1": "2da3df29b15785afc810624812ac29f2cc098afc4bd46269bdbf2a41e210616d",
    "s20307615-d7-b1": "b88a995a027b42a3a1b0a132a6518e82d5cda795dd73d346f28c1e8f3d51927e",
    "s20315534-d4-b1": "35e04b64ca538619d4b51c46f1fbc2f95f0713bff05511f362d27ada60724968",
    "s20315534-d5-b1": "0f6ae29f6ebe93cd510fa67aec6923225c446c810f80837a7e34e16c405e1707",
    "s20315534-d6-b1": "dd7f655d9dfa7f1c73aadf1973a38ca16201904f9874d1247fc9660a73aac33c",
    "s20315534-d7-b1": "d87e7ffb6f2db271c717f7936ed5dd5ae34c31295910217d23d2ca36110cf546",
    "s20323453-d4-b1": "26ebbb33ba9fc7c3af65cf2b8c7e53787f38fb3f266d1e6e685b71e504d323d8",
    "s20323453-d5-b1": "4f5f0708c0980ed6961c8b696179ea8531c728e78aa174413341aba5c07f179b",
    "s20323453-d6-b1": "79c0708733bff60b3b1ac2e5c65927dc553d43fc8fc168161a79f0ab35092181",
    "s20323453-d7-b1": "1c814bcd75fe91912839676ed18546994ebbe5b14f05fdf1ed01f58c9d90a41e",
    "s20331372-d4-b1": "ed3371b57b9b9a6580513e0bc8ee8acff673deb671dcc6cbc659c66861f06a24",
    "s20331372-d5-b1": "7d7029c3cd0ffa6a5d2b56c8f93849036426fab2d0984481754805a88e2aa457",
    "s20331372-d6-b1": "82a324c09dc7f6e538b65c4a688d8f5a274513c523e11b518f83b36e016711ca",
    "s20331372-d7-b1": "2b6d3d933329beaeff80f536eb637d4ab7b0356ebf5ab08591dac26c7f833c05",
    "s20339291-d4-b1": "96406b0b1e71317ede4d51211e43387fb21bbfc887a28da176fa07b8b549572a",
    "s20339291-d5-b1": "abf8c0a9779fa4084157dfe726526f93173131cb09e001b544108f6b19ed7db4",
    "s20339291-d6-b1": "f0ad0c695bb84563fc75f3bfbc8c440684d1df7cb41808e7e9eaa5ce2fd54f73",
    "s20339291-d7-b1": "5c1fca9940483e7df9b4acb1cccdb44d78e15c958b1895bb46adf54bf17ca5a0",
    "s20347210-d4-b1": "310cac7712d86b7a9d18c05d668e04d94a66f24206af6092fbb4a50ed4377593",
    "s20347210-d5-b1": "b6a3d0a465c165093acc2ba76135a3fd1f8862b0d67da48b0291134372233f5c",
    "s20347210-d6-b1": "e5f91164a2fb78fd393b5da6ef8633acc7a130757a863de97a82d6e2b83b3d62",
    "s20347210-d7-b1": "d64a7a1db0d802632b66a7752c361b74c19d0d56c803060e1657033081875a39",
    "s20260101-d4-b8": "b3f87229710acfbac3cd99b24a1742c77d83cc328f4129507fd3a5fdc8bf3375",
    "s20260101-d5-b8": "62f4758d5baede4f0cb29dd647b1709657367ed93e06284a713e469af7f6e6c5",
    "s20260101-d6-b8": "3f9f70edf6550af864a479f86b7c29bbc2412d68af03bd7ea31829ec5c7e5c71",
    "s20260101-d7-b8": "679954ce478746a081aa1199714b05d91094e8c65445f33148534deeee35f4c7",
    "s20268020-d4-b8": "950cfe0250a8c6d2f8b719f71aa55703d4dcc9088fd0c7a37b5dab99e126c400",
    "s20268020-d5-b8": "4cc897e3315de07f38e0240676e94549f1a3fd9dcd123d29873b4640870ab08e",
    "s20268020-d6-b8": "b83c47eefe8820bca3462bb66df6aeb1e70d274431c0c8272f0bd000e973aeab",
    "s20268020-d7-b8": "143dbda671201ae722980eb45d7b4f5a88d3c670de53aad2217de830330790b6",
    "s20275939-d4-b8": "35f4d8dac71e1e92ee0e19efdd424eef657463b87ccf23c9cc029aa6582ade80",
    "s20275939-d5-b8": "96d6ebe5072a5de4b47d3a9d9a24a39bb4b9ed1469af65df18d2fee1ce5b3697",
    "s20275939-d6-b8": "7135dd32b05667998bb25a1a619753e769027a48881817578c7f2ede637309ed",
    "s20275939-d7-b8": "dbc010f2633ef8ce22a0ba5b9447775c0cee290e56b92e4e47237268d96842af",
    "s20283858-d4-b8": "7f7a4aa0d51709c16da70b7fbf598802d0df32c9e0f650552a2de2ea17115737",
    "s20283858-d5-b8": "cf7acaa8d7435437b78a3778985193ed796aff99de4567d16e8d7e02beebcc81",
    "s20283858-d6-b8": "528aeeb91dd19e89d7cee0e827769bf8498eb0bfdac13315d38ca6c662cce446",
    "s20283858-d7-b8": "f9d8000a7de874acd6dd4764d563febe518a6053abc630e016df142bc4fd1e8f",
    "s20291777-d4-b8": "9c8e353a62d4d9cef96369553d1275fcf173acfffcf3a18041e629e820b0e377",
    "s20291777-d5-b8": "3244c8cb3f86050b750832710d8d2cc4f0ef7ff381f78b5ff368c839232b112f",
    "s20291777-d6-b8": "abfc802b1d0f6f4dade46ba3255fe4de921606eed0f169d784f19f5d67f9ff43",
    "s20291777-d7-b8": "8220cde43b54c758f6379329a91c105e2dc24ef6753f010199ed632f0674556e",
    "s20299696-d4-b8": "bf43871aabc3bf1ff5e5a6fbe7347d676089108c6ffda02557b9df1bac3d557b",
    "s20299696-d5-b8": "12f723f734b343b4ad1fe1dad01b399da18dc69a108661fa0ddbc7983c657f9c",
    "s20299696-d6-b8": "fca7a4e0d183dcb6c764703a4cdbd2e721e33004eda5e408469821053722e32e",
    "s20299696-d7-b8": "dc92b732a40ab41fd2cca94efb0c28d834fab65a3bce0624262baa80c16ae75f",
    "s20307615-d4-b8": "36a741f0081df99f8894be9d2a40f91566d7c5732c6f779d59e50eef22040042",
    "s20307615-d5-b8": "a22b15598ef03719b72fb8314c83144ab4395146655a1aa838d5c8b2a83dc7b2",
    "s20307615-d6-b8": "b3d272fe5aaec0a692f68a0f8b424f1d3b329acf1982538c47ddcc5e8b6c0c3f",
    "s20307615-d7-b8": "62d6c0944a3f1a82be8a54f43a5ff913d87be6e431d517b0c5ca91743eef71e5",
    "s20315534-d4-b8": "7cb425661ecbbd2153fe26f553195bce9a491f517774b7793669b65b3eb7db1e",
    "s20315534-d5-b8": "a4493e510c866031961ddf84ded142432ce11766d6c018ca7b134e03a85dabee",
    "s20315534-d6-b8": "ba1ace41bb27bfb4cafb269d04da7775ecb9525809ca9a447e4ca3c59b7be872",
    "s20315534-d7-b8": "86267123d66a1f0d42cf1f2d5e757bff55368a6e4a6a6127c0af3ae81d8b4da7",
    "s20323453-d4-b8": "f320b1918434fea0771623849eeb30c8a31468c1a759639c892f8d81aabdc167",
    "s20323453-d5-b8": "3834ea92887db58ba536f07a3e51e1a528519beef031de1c6a38ee02342b18fe",
    "s20323453-d6-b8": "cf08e093a386f9a0065cdb88f90dc1918d39d5c9a39853a87e859900c6cf4bdb",
    "s20323453-d7-b8": "91a0699ac06074182ad5a090bb075331b7337812886c049a49680012e174f74b",
    "s20331372-d4-b8": "40830547c92fa086254f45aa9c8a8e07c9a804a58aff02418cf4d18929cddd34",
    "s20331372-d5-b8": "54eebc021af3fc81ba3777e0acf38da5070c8c89aaa71c0be5770393958cc5d5",
    "s20331372-d6-b8": "9357c628911ec92ac2bff50c271306b10093ab84f303935c71d2329bfb3052ee",
    "s20331372-d7-b8": "ab570c6479bdc8f22adff3681ec6d0ae36bae52e1e58967231fab3f449567e93",
    "s20339291-d4-b8": "0d2c8c4af66624ac7eb9f0e3151134662d7915741c04e525f29c3033821b1e9a",
    "s20339291-d5-b8": "41829528e2b3d96eceefdecb315fd8e339ed5b525c912033a868d15a6f48e6d5",
    "s20339291-d6-b8": "98e94fb3473804fb2e78a5fa273b5ffb6859c640a1229ac7d45b786af3638c9d",
    "s20339291-d7-b8": "34c38d7b4185ac68af7964e8cc0e4db4d282ea559567b055054a6a915f03face",
    "s20347210-d4-b8": "7d5ae60a46d3bd7c898b3d1b7208e1706d599eb10971b762d075864d581badcc",
    "s20347210-d5-b8": "c3cdbc9a43b89a6d347dfe415d158d4633e3e9bff71f38dc288632d5a6df1c4d",
    "s20347210-d6-b8": "21ebbf8cb3cef65e91125e9f3789a68f191a752635c0773246075a69b5c1a853",
    "s20347210-d7-b8": "4f3bcefb4529775dde44db4badef651936204e72f805382de984224462dcce48",
    "s20260101-d4-b64": "b9458b12f83f3373a3115929cd24763f22df53101dc00099a3b5716021127666",
    "s20260101-d7-b64": "cef334c98a32b3347612177eeea647b78c32b959badf1006dbf33a43f9c90242",
    "s20268020-d4-b64": "8a92ece7c369435b2c9f50cec1daeff7ae7507023933811b3020d06184aa9773",
    "s20268020-d7-b64": "5c4bd6b12807ade99680a56d3bb0f3f60864e244c5bd7f53df1138beead15a99",
    "s20275939-d4-b64": "16086eaaf4fbb68fa1bc1f3f59a68d2323206c766d32316ddd40916b15ea041f",
    "s20275939-d7-b64": "a41c7c6c03c1354a087a551465ad9eef285e8e3befcdadfa5f8af1ccafe8dc4d",
}


def test_the_sweep_covers_the_enclosed_room_events():
    """The sweep's difficulties are the range the enclosed room's events draw, and its term poses the ceiling and
    builds the robot spawn pool, so every phase those arguments reach runs under the guard."""
    difficulty = _env_cfg._RANDOMIZE_PROC_ROOM_DIFFICULTY_ENCLOSED.params
    assert (difficulty["min_level"], difficulty["max_level"]) == (DIFFICULTIES[0], DIFFICULTIES[-1])
    params = _enclosed_room_params()
    assert params["ceiling_entity_name"] == "ceiling"
    assert params["robot_spawn_inflation_cells"] > 0
    assert params["placement"] is not None


def test_enclosed_room_output_matches_goldens():
    got = sweep_digests()
    assert set(got) == set(_SWEEP_GOLDEN), "sweep shape changed — goldens are stale"
    diverged = sorted(k for k in got if got[k] != _SWEEP_GOLDEN[k])
    assert not diverged, (
        f"{len(diverged)} of {len(got)} enclosed-room cases diverged from the frozen output; first: {diverged[:5]}"
    )
