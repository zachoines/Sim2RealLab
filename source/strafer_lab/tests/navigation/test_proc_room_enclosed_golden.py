# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen-output guard for the generator with the enclosed room's arguments, the room every depth stack trains on.

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
    "s20260101-d5-b1": "a205c727bebaa712fc2bdba069ce3022ac0fc8e5a3e80bed588190a46e4b91f5",
    "s20260101-d6-b1": "08e4fcbca25c24c1a354a7189def23dde9ae6648eb869f91ccd146eebfa2223c",
    "s20260101-d7-b1": "23478d45fcc4b5598ca4127fdc1b31e0d5b56442f36cb2dc98d2a93ce315ea88",
    "s20268020-d4-b1": "63f233c96eada2624dfe315230fe1814839bc153bbe2fcc4e0aee64a317a9f1e",
    "s20268020-d5-b1": "89cb8fe15cc0c978a01a63a46104e45202af6d72089127ea3a67acf55eeb7661",
    "s20268020-d6-b1": "10420dfa452ec8c2fcfe0b2faf8c1ddf467eabb3e4799808c879a722e457cf61",
    "s20268020-d7-b1": "90ab7b047b599173514a6a7e532452d6b70316e4a03c4d2dda17ed161f428aa9",
    "s20275939-d4-b1": "73b349856a17d2eb48f4d73370e80ec3a35e43c3d44b1c0269768baae1ae0453",
    "s20275939-d5-b1": "5ddfd8320653392660a8fc48aa1ac412e1fbffe0801751339251bbf7b500b5f5",
    "s20275939-d6-b1": "bc167d6dcef2ebec17350541691196f04e518c0d89feb26e73af20a6d1a7fc97",
    "s20275939-d7-b1": "f80339969fb28170c3dace5673afcfdd38341b870ad901e2ea7cfe1d33b6a79c",
    "s20283858-d4-b1": "335a4b57494b53a82f7f465207fe3e3adb88cfc05c3d0734e24528fff929cfb8",
    "s20283858-d5-b1": "d27b600bf928568cfb46be9fba8de4be15ad88fe8c01de8af902e974ef761a3c",
    "s20283858-d6-b1": "0f84d02dbbeb268b5a6ade5e5fab520f8242855e941f1197d9275967ed79a898",
    "s20283858-d7-b1": "25900b8eb3882d4a61863238b8ba6e38c280536634aff379d6241e03a98b24b9",
    "s20291777-d4-b1": "4f363c6b26e7f2603d016dbb10dc70a86f78318912003f0d672fd93240ccfa5f",
    "s20291777-d5-b1": "76990b33460549899a82edb4c9d368c6c8cacaccb0d45e4ca0cb7147a9703ab1",
    "s20291777-d6-b1": "b3d3b1e10be112f06ff7c7e30ded61e0c156970aff555baeaf469ebc40bd5eda",
    "s20291777-d7-b1": "bf0ec5bf7f71886e4d0def053b4933e32e7443ee93425f8f3423b8a6d5a6bd68",
    "s20299696-d4-b1": "2f8a17ccacc2dc6941bf5a0384ffbe1cc4861b409ab199a384b6076586cbd326",
    "s20299696-d5-b1": "c75da2b62e886a11803ac0ab89679c05c51217a90f382d9c6257665fa2ac4d46",
    "s20299696-d6-b1": "5c23a05797cae5c3acd80994089e6a80bffd1e70208df4c00bcc694e40cd636d",
    "s20299696-d7-b1": "6fac8f8f5a8dfa965d1b1a21914257e6723cc882ff51d3ef0ee9748af99e09a6",
    "s20307615-d4-b1": "2ab80e12a5c5f2102af4ac7d97812df87a806cda8d7d8cd7b8a144e0265fedb6",
    "s20307615-d5-b1": "e33bd7715a5ee85c007744b9d33c8efaad32c769ba79efc5d4ee70f40e8b962f",
    "s20307615-d6-b1": "5e960f4b08c208a5c94a07de0abf93f0322b54d0630945b3437551ad7044865c",
    "s20307615-d7-b1": "7a48392b3a8632855835265d7ed8947aa9b44fe73a3839f52c12c3c3b1ce64db",
    "s20315534-d4-b1": "35e04b64ca538619d4b51c46f1fbc2f95f0713bff05511f362d27ada60724968",
    "s20315534-d5-b1": "0f6ae29f6ebe93cd510fa67aec6923225c446c810f80837a7e34e16c405e1707",
    "s20315534-d6-b1": "dd7f655d9dfa7f1c73aadf1973a38ca16201904f9874d1247fc9660a73aac33c",
    "s20315534-d7-b1": "c8fea5435c6aa13816c1342fad93eb87ebaa4c1caa1d595694f1caf31753ee2f",
    "s20323453-d4-b1": "26ebbb33ba9fc7c3af65cf2b8c7e53787f38fb3f266d1e6e685b71e504d323d8",
    "s20323453-d5-b1": "59282c00299f389be1ef2a0bfee7520e62731587fa914cd4ca5034daf6986e5a",
    "s20323453-d6-b1": "bf934b22f5d78273a2afa460a58b98bf55ca0ec4a3e26a2c57c4c52faf89c246",
    "s20323453-d7-b1": "ff6dc24b6fceefdfd1a7f0f14fae07eb479b73dae4355ea5785514b575c59132",
    "s20331372-d4-b1": "8ea523ddc3d43b3c7364d59b8f2665349dfd15687a3f85c04d329354aac8b275",
    "s20331372-d5-b1": "b7979cbaea77e99ba0df259ef9842f6ae30942735828735cc3a82320852f93c4",
    "s20331372-d6-b1": "ff2d52ce52b2bfdf60105a09867368258e1f7946f905d3e7c1c6745ee418926e",
    "s20331372-d7-b1": "e54417c87bdaa906bc499a9856d058f16ffff7c8a528055996e914a74235ee22",
    "s20339291-d4-b1": "96406b0b1e71317ede4d51211e43387fb21bbfc887a28da176fa07b8b549572a",
    "s20339291-d5-b1": "abf8c0a9779fa4084157dfe726526f93173131cb09e001b544108f6b19ed7db4",
    "s20339291-d6-b1": "321c76f1012542dc999d195fb89213f4aee47cd2b0873d4906afdad169f1dd5a",
    "s20339291-d7-b1": "8e3a56187322b4cd91b9e025b13ff08ab61a6bceeeb172c8ba02f5c9d26e014c",
    "s20347210-d4-b1": "310cac7712d86b7a9d18c05d668e04d94a66f24206af6092fbb4a50ed4377593",
    "s20347210-d5-b1": "b6a3d0a465c165093acc2ba76135a3fd1f8862b0d67da48b0291134372233f5c",
    "s20347210-d6-b1": "e5f91164a2fb78fd393b5da6ef8633acc7a130757a863de97a82d6e2b83b3d62",
    "s20347210-d7-b1": "88f868f7f37f0cc085aee65571b9d4a840cde240abe67b5c36e8e0f4835fbff4",
    "s20260101-d4-b8": "7e38a4286fa4b0655b078d816808b0f7cbe09cbf8a4c2b3d3fd212c0a7d052c7",
    "s20260101-d5-b8": "5618d2991b131ad58ed8137ae80cc3327adc0e98a39a7e7f261870d9402dbf47",
    "s20260101-d6-b8": "6f6c86f78d2b1d99971e4208ce43bcc826d17055e2a773d9856ce8408678a819",
    "s20260101-d7-b8": "615d7e6ddca5a12abc1c6c1dc57b77dedd64d9f11fe1b5f4d0e51bc11248b4b0",
    "s20268020-d4-b8": "fae35ed6878b493ed2b46885146eb026c9f5f6343d8ff2fc0cdb2baad2fd42b9",
    "s20268020-d5-b8": "7365629d057d991bbc8461538c7925515f1b6211f2b272a0cad4a089043124f8",
    "s20268020-d6-b8": "911d4d61ca3a6cf89944a155a7535e0fa637f32fc84c9d4b5709892e5057108a",
    "s20268020-d7-b8": "bcf1c706d511cad43da95c3a441d7ecccdc3da91ecf1b37a6e97bce4b6ec45e9",
    "s20275939-d4-b8": "7d548ed6a6db7488e93af3954f1bf9031c3b02f01c49885fbeca3423de9cc61e",
    "s20275939-d5-b8": "53b0f936023230587e70226e494168a6621a3fc43ab7c02289d6a57a5c3bf5df",
    "s20275939-d6-b8": "207b814d930a3f280fc2bb3a251407c08f73aae56f9e3f7ed770478e8a5cf500",
    "s20275939-d7-b8": "77824ce4045f92f4ca6e67a29ffa41c7bad5428ed81ea174013f4e6d6a7f66c8",
    "s20283858-d4-b8": "4d570ca69594186f3d66dd39d036f0f51140609bfb31d61386d9aa3a099540c7",
    "s20283858-d5-b8": "495bb565f7b6415c0c35ba74e1f304a7fe20bb583c53620d3e24efc4ecb86eaa",
    "s20283858-d6-b8": "95cbc0c4802009079ad067b0a6bc3dd6005f4594c7630e38a7d9e0f6998174e8",
    "s20283858-d7-b8": "9a6f46e77581f1889469e90f94d2ea2e4a646c283afe4ec903f31ecb9837aecf",
    "s20291777-d4-b8": "41f5af39d8d4ef39f7f9dc82c2d712c04be74084c68510a99940b66e600c6fbd",
    "s20291777-d5-b8": "cb45aecf9beffbf195d17f5c78d3a43031b35a0b7d3dc8b18539dd3a0435b6bb",
    "s20291777-d6-b8": "3f828f0fbca6405de633b0c2a142d9556004e8f02cb85ec2183137d80d8aaec4",
    "s20291777-d7-b8": "ec6c418350330478c72af40741e9d55d5b39b429880c3bfffbca7e70a415bd4c",
    "s20299696-d4-b8": "6f4087879b99229f809b347757df2c69b22d8c904bf6f80faf5bccc618f4cee2",
    "s20299696-d5-b8": "2a5660de54e3625c6ce535a0ee7a1c67420ba2c11dbff799e6e03f4699a59e1d",
    "s20299696-d6-b8": "071515454d38d80c64a2dda8b6ca6478becf1868359dfbc0bae369f9f4c95de1",
    "s20299696-d7-b8": "4cdb264ca7494ec6e7bb2c570a5641313e744ddc29f7719995cde0286ae8c4e2",
    "s20307615-d4-b8": "564955be70d729b8788fef96f76520cd0c7b8085d02ed9dcfc0b92c6e651f61f",
    "s20307615-d5-b8": "340d96da6a99213a8a664719ee80c16a98519fa3a9fc7cfe27ae3765755b0219",
    "s20307615-d6-b8": "32e56e4a85e4dfc3ef60396323b800f7f0e22637aa4b7eaf76a2180f3bfa875f",
    "s20307615-d7-b8": "a6e49dab3aec97ae101ca6b2ec8f1b01d5f0cf1fac20c3f37bf76cdc49eabdb6",
    "s20315534-d4-b8": "7cb425661ecbbd2153fe26f553195bce9a491f517774b7793669b65b3eb7db1e",
    "s20315534-d5-b8": "0933023b2099a1f3d3cb62793c5f867eae10c5bcb14922ef3002b228f04e9a36",
    "s20315534-d6-b8": "5c0da66ece38a1a2fe6eb65ee5304982f5633de18b5c41a4308a4160a6ae1888",
    "s20315534-d7-b8": "d29e9b31df69f9e1b197b3c5dd3e2bfd4c8551daebd1a46ce8e4236a6187ce20",
    "s20323453-d4-b8": "a5db803629c0e81e14374a5478b3489a9d9a470cc9de374092dedac95c6873a2",
    "s20323453-d5-b8": "6bb7cb0c534128191e4d887bad3b25ff74bdebcfc1e41ed020089b406f0dab66",
    "s20323453-d6-b8": "1f1370571165b17d66486d0ae812a26c13d9edad523e50f4dd062e83fbe3c225",
    "s20323453-d7-b8": "a6687f4e6c54efc9bce8a29c6fbb2d18076ceb942a3c4e391ce3ad379824ad52",
    "s20331372-d4-b8": "50100eb5fce86da1954e8338a2ad76fa7283d7e873dd098d9d123d81c9be4f3b",
    "s20331372-d5-b8": "8087ce2e7e45960bc0eb3a1c08ef25b299302deca710955aec92846c45c31879",
    "s20331372-d6-b8": "4ea8d66184079c4c72d00117cfeceee312ff414afc05c3fb2b7d7ca804fab030",
    "s20331372-d7-b8": "5daa2069ad3bd1a5d679da28c9562f75fe43534745b8e00a27f85931b8b7b27e",
    "s20339291-d4-b8": "084501387e1b585b2864dc36331390be42231773b92627460fc7032de2ed0fb3",
    "s20339291-d5-b8": "c99486817d04514f53f3603b268ed2a3650c5284bed852c725304a37725a24f9",
    "s20339291-d6-b8": "790d46be7fa1514a3029ae982b75ae544dfc5a4401f7b7390512f16a20543124",
    "s20339291-d7-b8": "bfa8bc281fc13e2058352e62e85a74c3d1ae59eada691f2ea81eece1878f9127",
    "s20347210-d4-b8": "122f7665af7ca9f55c2497133399b3b98bc9fd89319e79c79a0e107aca9b5762",
    "s20347210-d5-b8": "13825fe448c8fef54add0fbb4eee9f063bd3b03d7c78cb18d7c0a47f77696294",
    "s20347210-d6-b8": "6bdaf10b53ca519884ca26d2964187ac5fe00db3a1ab373d687db57466365218",
    "s20347210-d7-b8": "23ab7eae41a4cb1d15f14a5a7ca0f4b4e12d60241ef79188aada2649812983e3",
    "s20260101-d4-b64": "dca9872033ef2d6c1c849a7cfbd80033e7707e78e38b403bfa5f964fcb595528",
    "s20260101-d7-b64": "973a6cee67793682f671e1817be7117c2aafc7b7dd4d8c35d142ae70cc8bbd85",
    "s20268020-d4-b64": "45ea5433afc5836c5e959e2aef03bd018718cb4912384b46586254dde4c6cba8",
    "s20268020-d7-b64": "16eeb6c0aa1274fb32224a606d1f23f930882535f2aebc8f5bb89108ba9b85c5",
    "s20275939-d4-b64": "91bb0df7dc4cbe2c6b44ddb9b45541732dac061e860bf3bda43e107509df7c51",
    "s20275939-d7-b64": "ecd1b156f7dbee1263ffb3d79adde9af89e418a10e4ff6597a6fb397a4325e09",
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
