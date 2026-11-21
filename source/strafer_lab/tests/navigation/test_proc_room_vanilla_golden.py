# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen-output guard for the generator's default (vanilla / NOCAM) path.

Pins absolute values — an output digest and the RNG operation sequence. The
sibling suite's ``test_default_path_byte_identical_to_explicit_defaults``
compares two legs of the *same* checkout, so it cannot detect a change that
moves both legs together; the frozen manager hashes cover manager cfgs and
scene scalars, not the generator's geometry, defaults or RNG consumption.

Golden literals are the generator's output at its argument defaults, the
open-top room. A diff is a contract break, not a test to update.

No Kit, no GPU. Run with the pure-Python lab tests.
"""
from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from strafer_lab.tasks.navigation.mdp import proc_room as _proc_room


# ---------------------------------------------------------------------------
# Stub scene (mirrors the sibling enclosed-room suite's harness)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

SEEDS = (20260101, 20268020, 20275939, 20283858, 20291777, 20299696,
         20307615, 20315534, 20323453, 20331372, 20339291, 20347210)
DIFFICULTIES = tuple(range(8))

# Batch size is swept because every scalar draw is issued inside the per-env
# loop, so the stream is positionally coupled across envs. Batch 64 is the
# production depth-train width but costs ~2.3 s per dense case, so it spot-checks
# the open-field / sparse / dense regimes rather than the full grid.
CASES = tuple(
    (seed, difficulty, num_envs)
    for num_envs in (1, 8)
    for seed in SEEDS
    for difficulty in DIFFICULTIES
) + tuple(
    (seed, difficulty, 64)
    for seed in SEEDS[:3]
    for difficulty in (0, 4, 7)
)


def _feed(h, tensor):
    t = tensor.detach().cpu().contiguous()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())


def _case_digest(module, seed, difficulty, num_envs):
    """Digest one generator call's observable output plus its stream position.

    The trailing draw pins RNG *consumption*, which is data-dependent (rejection
    retries; ``randperm``'s size is the reachable-cell count) and so can change
    even when every pose matches.
    """
    torch.manual_seed(seed)
    env, entities = _make_env(num_envs, difficulty)
    module.generate_proc_room(env, torch.arange(num_envs))
    tail = torch.rand(4)

    h = hashlib.sha256()
    _feed(h, entities["room_primitives"].body_poses)
    _feed(h, env._proc_room_active_mask)
    _feed(h, env._proc_room_free_space)
    _feed(h, env._proc_room_spawn_pts)
    _feed(h, env._proc_room_spawn_count)
    _feed(h, tail)
    # The default path must never pose the ceiling slab or build a second pool.
    h.update(b"ceiling" if entities["ceiling"].root_pose is not None else b"no-ceiling")
    h.update(b"robot-pool" if hasattr(env, "_proc_room_robot_spawn_pts") else b"one-pool")
    return h.hexdigest()


def sweep_digests(module=_proc_room):
    """{case-key: digest} over the sweep; ``module`` is a parameter so the same
    code can drive a historical copy of the generator when goldens are captured."""
    return {
        f"s{seed}-d{difficulty}-b{num_envs}": _case_digest(
            module, seed, difficulty, num_envs
        )
        for seed, difficulty, num_envs in CASES
    }


_SWEEP_GOLDEN = {
    "s20260101-d0-b1": "8e48346818190946d47738e192fd5960718a8a62d87f2bd79713bfe60dbe2456",
    "s20260101-d1-b1": "73b1f7d2ef731512a61fdd25b07a6d201dcb5b1fb1b06b20945636423aad9d83",
    "s20260101-d2-b1": "b8f9617b90a6ea92fac7f19ea0bb4dcde05271564398347b224869c8f370229b",
    "s20260101-d3-b1": "fab2d498520c1368eb2ee0b27c8f66169654710bb4f5ad0feea94bdea2472117",
    "s20260101-d4-b1": "a2c7307cef0a35f019ed2d9f027cf01d3e80a028d9ab65a1823652735a456f86",
    "s20260101-d5-b1": "5385a2981949bbdf0e1408dde8746e4a3f3f68103bd57ee5392d5bca901e38f2",
    "s20260101-d6-b1": "b134176f3e6a86e80136f1c5693c39e1e633fb84adbc2ee81444499fc45e0e83",
    "s20260101-d7-b1": "4d784b818df5d980c0432ebdda8c9cfd4fd8337cbaa3187623ce89da8b04035e",
    "s20268020-d0-b1": "3db0d9f94dd8a4b1423748b323cfe416e32d0141297ff917c43788c85f93dc77",
    "s20268020-d1-b1": "8b82cfe71232ab9c7c18e4e75d0f5c7dc95297a00ba2146def650bace287ab35",
    "s20268020-d2-b1": "e9e6aae3d18787c31c70bb134411d7ac948acbf9f6e59c61b69c44fcb83aa16c",
    "s20268020-d3-b1": "17c8d9386cbc02ece65fafecb64616ce6ee80385a25db1ac1b0a3519891c25cd",
    "s20268020-d4-b1": "e809556dd810c861763ed0f429da0c94c89b9d0be5fcbe19dd211da1c38144d9",
    "s20268020-d5-b1": "f818d83f9531b468a657c4866b85ba8189a3ab394b0c5b7d1020e075f4c446fd",
    "s20268020-d6-b1": "4860293233f384afe7c1a488d580ca15711a45bccb8655b907675ff2fa65a012",
    "s20268020-d7-b1": "32df4169024b8c09822ee6ee4c6c05c5bc3a507c50f59b7627122dd9bd56a257",
    "s20275939-d0-b1": "a98f6d44eb544e683264f080d55517396a46d5f18162440dd663de09c212c001",
    "s20275939-d1-b1": "26fd3978fe7f395d46e6b4a936041011ba6b40eea58b0c7b1b0ac9b61923cbac",
    "s20275939-d2-b1": "a57f6efe4c9fa3b55c70b73c2c343594151f588496831094a10363bcfa5fa9a9",
    "s20275939-d3-b1": "a075ade0c1ed2495131b6b236eea16357792385178e52425a7c493fc9050c4bd",
    "s20275939-d4-b1": "0e1e32957e61e423e5d950b3802f82ed3c394a5b04cb118f22e446350668f1ec",
    "s20275939-d5-b1": "7e5fd494ca79bbd8d7d4d29340624be747d30fb28c48cc7de6684602707a9d54",
    "s20275939-d6-b1": "26342faf5eeb39988ac3d047d642a04d03d16fe467ae375bc52c0d009942aefe",
    "s20275939-d7-b1": "4c3cb7c85daaf6204e807acd3c2e82dc80ffd6f048180d88761f1e102c28a5f6",
    "s20283858-d0-b1": "39fa5390be2bbab67ff999f7c2c1a09fd15086e629d3ea9186fb664ac05fa699",
    "s20283858-d1-b1": "15941a23f41c8e079718c5b3e769e4be2e6a349ce3f1b1acc70fa03a60b1862e",
    "s20283858-d2-b1": "b11d41a81598c425ec518d875963f67040ea17f5fad86d46813e5a5b53f403fd",
    "s20283858-d3-b1": "3aa177ff7c8da3f470e2e7c6ad7bed905bbb9a8ecd6a8534ffbff185a27e809d",
    "s20283858-d4-b1": "78f6a05b6c4eabd904f54c5bd5d5e926c79dbbad294a2ee174a564d071096bd0",
    "s20283858-d5-b1": "c51775908c40f57d2a4633dca19138c0a1e68daa4dc4d3985c50067ba1931d2c",
    "s20283858-d6-b1": "11991bd32e5e592c362a98337eca12f03e5cbfb4c83e89178ec67df2f4191749",
    "s20283858-d7-b1": "78a4f9d39255fad413c144c96efc422b9da71dd30736914162e4b04086e13b51",
    "s20291777-d0-b1": "6e6f341b7a86cdc7e0351f369789ee61ee9ddfc1bb852839690d9c1b1d0560bf",
    "s20291777-d1-b1": "f88ac8f376b6dc45677bacb5696a5a5137b0f769cbc03737fd4376bd5e2c681d",
    "s20291777-d2-b1": "e91e7bb2b831a0b5c37b8435d6657120fc53273aa81894e77707ec55b94c4d6e",
    "s20291777-d3-b1": "71afd75fb5e8d585b2d98815aca037503c275d094a54ea0f5927a876a0f0d9e2",
    "s20291777-d4-b1": "b293bf67e039f15f1a2627156b19e0ac4412ade9014a578bfe978868ea135184",
    "s20291777-d5-b1": "57954dc0fa943cee10ab1bf195a0cc872557a51af34ee5a116ad6eb5660e660e",
    "s20291777-d6-b1": "4f31eb6770aed37ed28f8de07dca43dfea2d078c56ad7041888f23d0a1fba9bb",
    "s20291777-d7-b1": "cdd7ed37b0781a6e503cd69cf7da460268e507103a6b69fa218898855aeb1a17",
    "s20299696-d0-b1": "6d608bd608b2cd20c4d93ab0bc151df0cc34bd8455acfb7dcae8f07e5a8ef02a",
    "s20299696-d1-b1": "ce134fd3153b7fbe997d83e2daaab8cd3e04aa738b1a5855074ffc8f75ce572e",
    "s20299696-d2-b1": "6e42e84b0d9e853b0fca1a7366765dc7886c7ef72ae0adfed1fb79a311c534fd",
    "s20299696-d3-b1": "60eda8d30be3df8497f6f338202b4443ca43c2972cd6d9df4beb8dd09935c1eb",
    "s20299696-d4-b1": "1d916eef19983c0cd8d23dff33ec946c65679fb65b78510760434b7328b4b29c",
    "s20299696-d5-b1": "9894932ee3d3f8d18b45c0d0d430b3bb2700e25a1c4b56be76130e75b16fcdf5",
    "s20299696-d6-b1": "c5c3fa5e4662ddb1a9ce20b72507b730192a1de4761ff98445f3cf6d67d79984",
    "s20299696-d7-b1": "9a057dd945c56f8cef9fc80149947d3ace7fe21ba7fd8f55c3f6ddebf2f4b201",
    "s20307615-d0-b1": "83ccf36a7322386540bc7acd8a5decb65d3ef862f35ec3f4f4fb17d74385cba7",
    "s20307615-d1-b1": "94d1a19acfff6bb8e893154a868b6893131af4d4003fc972ac3da57b4896401f",
    "s20307615-d2-b1": "47602d2a0a0435eaecf8efd647f7a4e6e8dad69aabd970bb2e8a2ee72aa1579b",
    "s20307615-d3-b1": "a6a4284814269942164b51c387e120265b192f36e1938dad3ddae52e594606e7",
    "s20307615-d4-b1": "982754bd6e45728dab9fae3110268aacc26358ea81fb9ba043243576817d0453",
    "s20307615-d5-b1": "29f4b8648e5d4a70003627d17360ebd1c8eeeb02321afd69361bb16363f75389",
    "s20307615-d6-b1": "e377983f8d3623bb90297112eda2c1789bf392265658cef499cfb297fdd9ba95",
    "s20307615-d7-b1": "ce21166e1ffe5a7050cac6c33d10fb5d40bf04398d998125021f227af77122b4",
    "s20315534-d0-b1": "1baf901c3ce927a69692a86f133d83e8a724375e7ee2f079d1e5f5b0b7b900e0",
    "s20315534-d1-b1": "f1ca22e9dfc5848319cf74a5183cfd880e3bffacaaaff94124106411657ba27a",
    "s20315534-d2-b1": "a7431fbf7a9748e4e6601bbc22f9899b80c73a0dbf681234061d959c9977318f",
    "s20315534-d3-b1": "bb97e0bbe675f68859e4f7a31cb68bb2adf7c729d11c13ac04af7d7557aa269c",
    "s20315534-d4-b1": "86488c8db2a8f54fda144aa0951556ee138aed26df1f582d395d44735aae5d06",
    "s20315534-d5-b1": "9db0ada56ab1203815794aae54a23bdce748978d61da04fa19a37f6b630aca3c",
    "s20315534-d6-b1": "f930bae49a0c5d6d9291253c7208a8f1e6644ea6234ac27bbe8a9c695c39f3b4",
    "s20315534-d7-b1": "ca78b960f1b7e83e8d779655140eaf47254b0c6812036d28603272a74651f0a8",
    "s20323453-d0-b1": "0c04439764ef5c7e20d83dff06753701f41118bff943fb19dc0a55790ebbba35",
    "s20323453-d1-b1": "f982bc0650002b9788eeab49f794016358f461cf416c8458c9e0a1b2c45915a4",
    "s20323453-d2-b1": "e8725aa9ca0689a1167b3184b61ec4d549e12fa3eacb4f1dfb324744413ca96d",
    "s20323453-d3-b1": "6850a4649cb9ff995de488d3757ffadec147640cd01ff1071375e674d3f3eac3",
    "s20323453-d4-b1": "5232aad149e8073ea91c3126264204383bff922c0d797883efde0d6f58a41f5e",
    "s20323453-d5-b1": "7bde23670ed2d0e7b1aab4a533cb27276b5865177096fd46c55826af0e6499e3",
    "s20323453-d6-b1": "041a0116f6c0bd2444559b438897e498b3bf38d926fca5a4f8c38d656200c347",
    "s20323453-d7-b1": "65f265727fe55059d7cf037a8a51be7d5cbed4a8746cae742961b06e3e86080f",
    "s20331372-d0-b1": "0335d2ed5ab321d90a7f1ea74525f6c69a2bbfc421787472b21126f4eaf8f5c5",
    "s20331372-d1-b1": "00226920132f6bd1e90f3f7f22c76cceaae1f411bc0a4b9dfa7564efc631a787",
    "s20331372-d2-b1": "250600d9a8214beb68528f655c9639dcd59cb0d25972ba27f5bead2475b28460",
    "s20331372-d3-b1": "d3a9187a584a53d4cbad11e8f9f6dc5fe50b4f75267cc242c46c8e4e71776779",
    "s20331372-d4-b1": "866a68209d0de424f25f5d20dd962fb8da92bb311bc4e29a78be9aec8689e6ec",
    "s20331372-d5-b1": "203975f01e8e786ce926fb1f819652ee2e93e5104daec51aaf5579e28d49c0d7",
    "s20331372-d6-b1": "caca9e3acf955502735cae1bb73e6324051137038ec771981a37034fc5f379e0",
    "s20331372-d7-b1": "18a83d8e69942b808631ca985bdb627db77ae72087fbc92d3733ea57dfccaad2",
    "s20339291-d0-b1": "632f6eae0bfa660184389df94cc289ad6f8983497e8fbc502df035c72781e9fb",
    "s20339291-d1-b1": "7559d36c1423e4fbf86e1558295b360d8d4d4e546f88f4a93bc1d6864ffbb48a",
    "s20339291-d2-b1": "1e1dcce39f19862b829df5712990fdab927657a8277d9ac4194d5b6ba39250f2",
    "s20339291-d3-b1": "63ccb96e2689553200bed46c931a44a429fff6b7557a55c97dc622d725082787",
    "s20339291-d4-b1": "0dd9cfec8d0a6cc5772039a28211790ed42c23d99880e46b17f1d4a6060a52de",
    "s20339291-d5-b1": "9edaf35e546d563e552699d76dce90f692e909adca423485122ab4369ed9a200",
    "s20339291-d6-b1": "e7f46cfff4381c71a4e33b2e2a0775d29148ed535502ba9bb2a3596d8f028740",
    "s20339291-d7-b1": "73181b436f20fa4eb5e833881d3ef172a058d6554c7e8904c8ed50d110f13c61",
    "s20347210-d0-b1": "82cc66e9d989d728ddf31020ca01dad07ce02c0c272c00e449352ddae0234241",
    "s20347210-d1-b1": "36857015196968a08fb9591be73240ee6b00fd24e10f65c490dc1c4418b503dc",
    "s20347210-d2-b1": "367042b8c92480e280f7c857f9b1346187568d30f6847797f88cb64910f9a4f1",
    "s20347210-d3-b1": "276bb306fa23d529cfa4a454e1df053341ac5d65750627d1ac2b2e3c716a89f1",
    "s20347210-d4-b1": "0b34789156091144491c5e86c5c4e3bc6ce9db99d0789e4eed7abe7c73e121d6",
    "s20347210-d5-b1": "c6ad85096db0d26dccdb208e07582c0b0890621e2edd4a861ef125fc44199b17",
    "s20347210-d6-b1": "d18ba1b86afa097a75d289399759bea95c8becaa81320d01f1f7daa082cdacf6",
    "s20347210-d7-b1": "7688f8c8e3110f13ac36dc381c82b51ea0fa0e8df305886e878e837d47e25ac1",
    "s20260101-d0-b8": "ac683efe2de4d8d6d3d2b10fb100508e2d215c619cb723f19fcf51618e837aaf",
    "s20260101-d1-b8": "f0463070ddcd67fdf034203689e1cb1d0efa2cb95f562f37f9cba3d4c347adf3",
    "s20260101-d2-b8": "91629c7bfd9687924cf10e2247e1cd897c3b376ab8f1d68056499d55577ade63",
    "s20260101-d3-b8": "4e90113a51491524dbd7d08baaa4c47a1cb840ae42c63f78acfa4a8a46c934cd",
    "s20260101-d4-b8": "2d6420cba661110fd476ddca8d90121f796206d21ee7087da60fc67f40c742ee",
    "s20260101-d5-b8": "224356acee5771bc664071389e5d6cc611e46deae332920ef69b38513b89e969",
    "s20260101-d6-b8": "6df9650a8679feccb98d1338565f3f97ae46bf2f0ff174ca566a93ed84a10a46",
    "s20260101-d7-b8": "24900b0700ba90bd2ca1fda1bf84db95e621210bc39e8d19a86ebd00f3c1cf9d",
    "s20268020-d0-b8": "a194d0bf054824534e2f056cdd70f5ca912faa6c740764f864e41b7f5040cfce",
    "s20268020-d1-b8": "f494eed3850f59f9393660e89d045daa4ad5e4c2b82937ee262dbae8ff12cf80",
    "s20268020-d2-b8": "912c4d9a19dffb560a7be1e70e1c39094d36e59622c71e9de7b6c4edb1446f96",
    "s20268020-d3-b8": "80b760c7afd9c8344fdb31c1da260e99d8af1a059479167894216886819f26ec",
    "s20268020-d4-b8": "f4f7fa94942345ef27b806ff4342631bced498e7697402c433448cec7f48fc7c",
    "s20268020-d5-b8": "7b1e28076c8236ad65d71a20c92ec0f7cf5581f740c31b829421e7f671b450c5",
    "s20268020-d6-b8": "623c753c8f8829b86bd381d2394b0753c14f5899b2ea4555f8c367930883e19d",
    "s20268020-d7-b8": "b3dfbf6f665a0177685139ae27d5100ed23b235630058d836082a331364187c5",
    "s20275939-d0-b8": "69ed4c20a42b54e1bf297fc21a256a08fead26fed7f27c4b69acc96eff5eb418",
    "s20275939-d1-b8": "2c84d6f9c854b7411db1bcce57b21fb901d431835ec1629927adc2d6d98fbafc",
    "s20275939-d2-b8": "74a01c3aff6ce5b7b1c2efd0f2ad9db02aa30cbbdddccfbcf35eb86e5d3cefef",
    "s20275939-d3-b8": "32e26526c8a42d501508f3bbb607ebb72d9c844112f26a69271b1ac802a6e7f1",
    "s20275939-d4-b8": "8eae7f98aab916c144b16c89582ba62e33fdd8c03253430c0f155ac85344667f",
    "s20275939-d5-b8": "e310686a32f736a2d057f9b1ee7bd4b6e02a77a0be42a926a9031a0fa6e45158",
    "s20275939-d6-b8": "479d757147678eea0095e386b0be3d1e3772f687ded07412e1dc457af6c5a059",
    "s20275939-d7-b8": "9c1020f8eebfe7d00f2a6d48dbd36ac725c41750bd5989a490644679442c7619",
    "s20283858-d0-b8": "9dddff61c94a98a91ea7b9ffcba0f72fa13c0dac34127a169298c0e0c3902839",
    "s20283858-d1-b8": "042080c884bb59f739fdf45ce6a49a591ce350ea40d460fc45ca4fe8f9fcd5dc",
    "s20283858-d2-b8": "ead4808d29f5a659ccdb6ea5fa83fe45c1e1be1152f8cf13bce67e9db2063fd9",
    "s20283858-d3-b8": "896b74208813265629f4a6ab643f17a3610949fb732193162a39112a7b59211a",
    "s20283858-d4-b8": "f1f1bef76b4e1706a87adec691b1c4a07262843e4bcdff9f9002e3ce5e821060",
    "s20283858-d5-b8": "c6a5f5acd1cab2acfed5a9aaebf4fb10eab8a5b086e8a8f36527cac7b90ea1ee",
    "s20283858-d6-b8": "4374cce53894ee8490ce432a94fefdd0a61addede250aa766a5cc0d0fb187eb5",
    "s20283858-d7-b8": "3b04a3d0e98f1f6adeaedc1f1c0deb452673232d3d3ad67d6c1f67e543d98080",
    "s20291777-d0-b8": "6d4bdcbb67650433d16b47fb3ad4f6d67ad6e9894dfd416551b0dff2a9215bac",
    "s20291777-d1-b8": "74e72bf2c649c5882218960e0725f7eeb0f52b8b6c2c99cb6644a27a5816c5fc",
    "s20291777-d2-b8": "ceb9e998bd426345943523877ecf478b396d016530db620fc540e6b4c106e03e",
    "s20291777-d3-b8": "0645b5f0a7daf10aadcfb6c83f1967760f41500f6dd6c19afcd90945e14fa9a2",
    "s20291777-d4-b8": "becbc18d480a5ea8bb1c31e9d6d056dffdd4e8636d5ace1d9f97a005c44b5dbb",
    "s20291777-d5-b8": "3e35ff5a5c0a3f6ebfc4fee5ac964399197b0926b605115a21488e7a3db42b31",
    "s20291777-d6-b8": "e318385248d7c08a69665f7f8dd83637083f34357de879d79c479ec4b7c5b0c1",
    "s20291777-d7-b8": "9dd7eb5debacb10ffe0cc3b8960c0e7b60785ac232ed57077a6356344c70062b",
    "s20299696-d0-b8": "c1854308eb940972851a064d409f3f744b8e3c18c7c9b39d175c3192379c2c3e",
    "s20299696-d1-b8": "34582c70a4f99456425c95bb2e3e9f257965f73e94d3c709743956630f54ddc4",
    "s20299696-d2-b8": "20ff1756ef985f72fc6d19584b39a1827b2427d6304a82105b4aa7bf7d0f0efc",
    "s20299696-d3-b8": "53012338bbfd1fdfa0127097b41907a1e5a88c8747ab1535a40088eadadf1a19",
    "s20299696-d4-b8": "a1fa49ff2077d2ea128b7c997ffd62c704eb99826b3f9d6734c71cdf53893356",
    "s20299696-d5-b8": "6682130445323e1669f23ddec9ed3e31ac9529a37af17137fc62878b64b74ec7",
    "s20299696-d6-b8": "04f3fdea0ee45afa7f169786a9dd1e710d9c5b42468f54a2148f9901d57149a6",
    "s20299696-d7-b8": "430b556ffb053ec5807d0fc33efc0bf5f331d322264ffb3d4fb65543abd74447",
    "s20307615-d0-b8": "31d77107698062b956f35b658162297547f4f971f2ebb1dd95e6e4c00899a1b7",
    "s20307615-d1-b8": "a98b884175a4aaebf396564b83bb3446cf04d64589695d85bea20fb10fee14d1",
    "s20307615-d2-b8": "d27d6576e98337483c2f92959652feeab65f4be952630b101bb3d58c6735e578",
    "s20307615-d3-b8": "a55c4c77f1b9d6fb21c724af5379c52c482b8f25743b6282fcbca1c3c41e481a",
    "s20307615-d4-b8": "0e3544bfc61cacc83e48c6b36a7e21ed13a2b79e7537695c03c1ba91ed946333",
    "s20307615-d5-b8": "976047b538c7f9b8f62439b25c9bc94068fb01cf1f5f7ee23c0d71a8274bbf7f",
    "s20307615-d6-b8": "e612ee7650bdc146ecb3f8f4d4cedd37ac7094a30e6b6a48124defb5da00ce11",
    "s20307615-d7-b8": "38f3accd6749e8a5d013a9b84298940f5eaaa1b2fa9cf410ea9c5ed70a8be109",
    "s20315534-d0-b8": "3097e1b0847e22f5b628331b92cae09efdc99dc7d9be425169fd372fae065050",
    "s20315534-d1-b8": "73883646eca922bd433fca217e308a08f9fc0b362aa9416e5179ca75a9ef712a",
    "s20315534-d2-b8": "757c3ffce17bebd119b3383c167e8b2c261eee22acd28d4d8e6087a8c4c98a32",
    "s20315534-d3-b8": "113943bc3e503730b40206f277d6b5dff5007544cfcdfd76c37f72a4bf5f1518",
    "s20315534-d4-b8": "c7d33ea9e5288a15a4764161d9470519ae5b19675c39b21b065afc8ccd6ab519",
    "s20315534-d5-b8": "7ae29ba6c5e8c34a63e9697b3e3e78d5bf0b5ef9a1b395164a3340cd8fb572cd",
    "s20315534-d6-b8": "e567dd77d48ec1e6f32e57b3e48b158768fb294282554c8fda816bc46b6d9df5",
    "s20315534-d7-b8": "17ce9f3931b207a52a4c740c1649d66e2a5a9a314d55098423b1eecac1101673",
    "s20323453-d0-b8": "9c7ae8c2877251d6cd2bae8b19062c224e0f883ac5c42ec57b44e12ed9c298cf",
    "s20323453-d1-b8": "2e71cba8a52c3f2b2d46e94f76ae573f69c88cfd89043d432d0b03cd04db9237",
    "s20323453-d2-b8": "b88f3b4436b4f0c097b5f815d6c6ebe0a7b480b8125ca6d921d1e7e4621bbb38",
    "s20323453-d3-b8": "632290fe03bfdd29f8fd28654a12432ecdc1af1a8cad2af17067ddd11c7a2e8c",
    "s20323453-d4-b8": "fe7c30b6eca61967183ed91161986057d846d1e3a3076816597ad80a33710175",
    "s20323453-d5-b8": "8ba51cd2b50062387c08e50f2e1c3e191db31fae4279505c83b9bcd9df9fc4c6",
    "s20323453-d6-b8": "5297156021d7ad2c6d8290d6e3178d2561a8b9db321ab610946457de3e478c6f",
    "s20323453-d7-b8": "8ef8cc05be48d50af73652ed7711d7ea577d6f9295f499b48f35ce4daf838b5c",
    "s20331372-d0-b8": "85cec68a9b7c467e7b9744764080d83ebf02299498fe22be743331ee6e3b69b5",
    "s20331372-d1-b8": "46b0c8872bd0e92cb255fdd9ba5c742398843841ff7b1f493f2bdf215cfb4392",
    "s20331372-d2-b8": "f822157acd7a173e752aac2da6da2b7b808973331098b90da09a6a8f02abddfe",
    "s20331372-d3-b8": "313693f39742972cb0f608fec04a08ff5d13d31d2c0a8fe31bbf1d0e271088de",
    "s20331372-d4-b8": "dcec7c3426afb7b6e52db210f972394a872da0e584ec5e82ece3c5b36f3327fb",
    "s20331372-d5-b8": "b96b35dc90ac1cb47a291a1759c0efb363569227f061224b2abcec707a152289",
    "s20331372-d6-b8": "7a6899a4c4d8b3d83e042ef8ce292a3176feec22165a1f13f9d8df2aae6b310f",
    "s20331372-d7-b8": "014436853638a90852db2f51eeb29418466466f640ac30c4212935f808e2ff5b",
    "s20339291-d0-b8": "9fd9d0bb5c46ba426ffc0537af2b0e1eb6adc97c1a2418756ec3ea7ba7c21d11",
    "s20339291-d1-b8": "9c26145263ae52a658f120791b79816545caa39d81c60a7476579a99d15bd062",
    "s20339291-d2-b8": "0e437a2cba1e22336b9f4a5eac1775877a2f8519843eba5b9bea8f165484cf1f",
    "s20339291-d3-b8": "d9554ca7bf1d09476a1342d4f3c1a00822e2b439236657e49bc0555f9a66aada",
    "s20339291-d4-b8": "759d8941cea75d8a81d80fcd4ac2f2b48840ad3302e9b14bcfba01b75914f4d2",
    "s20339291-d5-b8": "86301daaddd746f4d7bb1fa71eea4fe609d3be214e5c26bcc097226f14de6f6a",
    "s20339291-d6-b8": "7315177f28810ec2ab6c888a34b564720d5549c7eac9c62e6e516647e0d9eb93",
    "s20339291-d7-b8": "57bf9cf0932091ab16bffcb03411df07bf97f359935c67b3b5b3e5779cee1fa6",
    "s20347210-d0-b8": "75a66e64903bd3cbfcd572b36d06f4868b76ceebf33fd87a49cdfb7e250ea0e7",
    "s20347210-d1-b8": "9de4cf9a51e291ef510be2dc589f38c47e6ce615cda376485d0dd13be647560c",
    "s20347210-d2-b8": "4cf80d9babf05be25b361ece653bce926c4f9ef07ceef7e19dde2aa7f49360a1",
    "s20347210-d3-b8": "f06e8868fd8af71159f31755b82874a80a1e28175ff79279bd6a362f468b970d",
    "s20347210-d4-b8": "f0b04b0c8a05882764af8104e2d8c26841770f65e7ba04da44d2c30f40be52bc",
    "s20347210-d5-b8": "1c9af7cf73af3d604f6cfa460bdd39c3504d48a957c660e9eb47fa354da0a1af",
    "s20347210-d6-b8": "7eadf861fb3c4908be138f6c56d6104426738d870aa858fb304951b8992ce675",
    "s20347210-d7-b8": "5210fb883f27452ad42df1c67ce22ae56df14ec875417c296f8ed7d471575950",
    "s20260101-d0-b64": "ca1065bd1dce4cdde23cb8efda128a86858b6167cdc8376779278eb2100ac795",
    "s20260101-d4-b64": "c839794512151d01f623d497f5225b9ba2e62d7a348687e4a1216cc1628df21e",
    "s20260101-d7-b64": "cb438c3bd93a0ef5af01d19946787b73b136aab995e618144cbdabb21a7087d2",
    "s20268020-d0-b64": "2cbe46ae73a57f18e48d316271275e2c6c8d1d174c53783c98a4866e349e04cf",
    "s20268020-d4-b64": "e2ba223c856db67c05a267ebc6e407cff7ffff8b6768011d56c5da2fc563a615",
    "s20268020-d7-b64": "25bde9ac0e5be11490a45e9f2d7faab6162065117b047d00cb625bdaca65aadd",
    "s20275939-d0-b64": "7462f83464493ef3b90ad4448841de45b13b21750ec77345903fb71b1757f5de",
    "s20275939-d4-b64": "11918b1795102f583d0709ed1c57cc4aee1b72c16d1ee6cbf2e75b5137a0df43",
    "s20275939-d7-b64": "b8c1989bc25f13a2d8c2d84f740975ce49571d8763002076ea55a2a798036ea4",
}


def test_default_path_output_matches_open_top_goldens():
    got = sweep_digests()
    assert set(got) == set(_SWEEP_GOLDEN), "sweep shape changed — goldens are stale"
    diverged = sorted(k for k in got if got[k] != _SWEEP_GOLDEN[k])
    assert not diverged, (
        f"{len(diverged)} of {len(got)} default-path cases diverged from the frozen "
        f"open-top output; first: {diverged[:5]}"
    )


# ---------------------------------------------------------------------------
# RNG operation sequence
# ---------------------------------------------------------------------------


def record_draw_sequence(module=_proc_room, seed=20260101, difficulty=7, num_envs=4):
    """The ordered (op, shape) sequence the default path issues.

    Draws are identified by index, not by source line — the sequence has to
    survive a refactor of the code it guards.
    """
    log = []
    real = {name: getattr(torch, name) for name in ("rand", "randint", "randperm")}

    def wrap(name):
        fn = real[name]

        def inner(*args, **kwargs):
            out = fn(*args, **kwargs)
            log.append((name, tuple(out.shape)))
            return out

        return inner

    for name in real:
        setattr(torch, name, wrap(name))
    try:
        torch.manual_seed(seed)
        env, _ = _make_env(num_envs, difficulty)
        module.generate_proc_room(env, torch.arange(num_envs))
    finally:
        for name, fn in real.items():
            setattr(torch, name, fn)
    return log


# Run-length encoded (op, shape, repeat) — 340 raw entries otherwise.
_DRAW_SEQUENCE_RUNS = (
    ("rand", (4,), 2),
    ("randint", (1,), 1),
    ("rand", (1,), 2),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 59),
    ("randint", (1,), 1),
    ("rand", (1,), 2),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 61),
    ("randint", (1,), 1),
    ("rand", (1,), 2),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 57),
    ("randint", (1,), 1),
    ("rand", (1,), 2),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 1),
    ("randint", (1,), 1),
    ("rand", (1,), 63),
    ("randperm", (329,), 1),
    ("randperm", (1147,), 1),
    ("randperm", (391,), 1),
    ("randperm", (1123,), 1),
)


def _expand_runs():
    return [(op, shape) for op, shape, count in _DRAW_SEQUENCE_RUNS for _ in range(count)]


def test_default_path_draw_sequence_unchanged():
    got = [tuple(entry) for entry in record_draw_sequence()]
    expected = _expand_runs()
    if got != expected:
        idx = next(
            (i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
            min(len(got), len(expected)),
        )
        lo, hi = max(0, idx - 3), idx + 4
        pytest.fail(
            f"default-path RNG sequence changed at draw {idx} of "
            f"{len(expected)} (got {len(got)}).\n"
            f"  expected[{lo}:{hi}] = {expected[lo:hi]}\n"
            f"  got     [{lo}:{hi}] = {got[lo:hi]}"
        )


def test_draw_sequence_counts_are_stable_by_operation():
    """Per-operation totals, so an op swap reads clearly when the diff is long."""
    got = record_draw_sequence()
    counts = {}
    for op, shape in got:
        counts[op] = counts.get(op, 0) + 1
    expected = {}
    for op, shape in _expand_runs():
        expected[op] = expected.get(op, 0) + 1
    assert counts == expected


# ---------------------------------------------------------------------------
# Signature + module-constant pins
# ---------------------------------------------------------------------------

# The composition-contract hash renders a callable as its qualified name, so a
# changed default moves no frozen manager hash but silently re-points every
# caller that relies on omission.
_SIGNATURE_GOLDEN = {
    "collection_name": "'room_primitives'",
    "max_internal_walls": '0',
    "max_furniture": '0',
    "max_clutter": '0',
    "wall_height": '1.0',
    "door_width_max": '1.2',
    "span_max": '7.0',
    "max_span_sum": 'None',
    "clutter_wall_bias_prob": '0.0',
    "robot_spawn_inflation_cells": '0',
    "ceiling_entity_name": 'None',
    "p_ceil": '0.0',
    "ceiling_height_range": '(2.2, 2.9)',
    "tall_object_heights": 'None',
    "placement": 'None',
    "health_sink": 'None',
}


def test_generate_proc_room_defaults_pinned():
    params = inspect.signature(_proc_room.generate_proc_room).parameters
    got = {
        name: repr(p.default)
        for name, p in params.items()
        if p.default is not inspect.Parameter.empty
    }
    assert got == _SIGNATURE_GOLDEN


_PALETTE_ORDER_GOLDEN = "ee2da00d4215a20f65179a78473eda7a7b3a74b38cc1d9dd357f58b153d2e5fd"


def test_palette_insertion_order_pinned():
    """Insertion order is the collection body index that every slot range and
    ``OBJECT_SIZES`` row is written against; the contract's palette signature
    sorts by name and cannot see a reordering."""
    names = list(_proc_room.build_proc_room_collection_cfg().keys())
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    assert digest == _PALETTE_ORDER_GOLDEN, (
        "palette insertion order changed — slot ranges and OBJECT_SIZES rows "
        "are indexed by it"
    )


def test_slot_ranges_match_palette_prefixes():
    """The slot-range constants track the palette build order by hand."""
    names = list(_proc_room.build_proc_room_collection_cfg().keys())
    for slots, prefix in (
        (_proc_room.WALL_LONG_SLOTS, "wall_long_"),
        (_proc_room.WALL_MED_SLOTS, "wall_med_"),
        (_proc_room.WALL_SHORT_SLOTS, "wall_short_"),
        (_proc_room.FURNITURE_SHELF_SLOTS, "furn_shelf_"),
        (_proc_room.FURNITURE_CABINET_SLOTS, "furn_cabinet_"),
        (_proc_room.CLUTTER_TALL_CYL_SLOTS, "clutter_tall_cyl_"),
    ):
        for slot in slots:
            assert names[slot].startswith(prefix), (
                f"slot {slot} is {names[slot]!r}, expected a {prefix!r} entry"
            )


@pytest.fixture(autouse=True)
def _object_sizes_immutable():
    """``OBJECT_SIZES.to(device)`` returns the module tensor itself on CPU, and
    the proximity reward imports it, so an in-place write would reach training."""
    before = hashlib.sha256(_proc_room.OBJECT_SIZES.numpy().tobytes()).hexdigest()
    yield
    after = hashlib.sha256(_proc_room.OBJECT_SIZES.numpy().tobytes()).hexdigest()
    assert after == before, "OBJECT_SIZES was mutated during the test"


# ---------------------------------------------------------------------------
# Ladder fast-out
# ---------------------------------------------------------------------------


def test_solvable_rooms_rasterize_occupancy_once(monkeypatch):
    """With no failing env the ladder must not run — losing the fast-out is a
    silent per-reset cost at 256 envs."""
    calls = []
    real = _proc_room._build_occupancy_grid

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_proc_room, "_build_occupancy_grid", counting)

    torch.manual_seed(20260101)
    env, _ = _make_env(8, difficulty=4)  # sparse rooms: BFS succeeds first try
    _proc_room.generate_proc_room(env, torch.arange(8))
    assert len(calls) == 1, f"ladder ran on solvable rooms ({len(calls)} rasters)"
