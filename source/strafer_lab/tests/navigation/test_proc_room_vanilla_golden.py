# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen-output guard for the generator's default (vanilla / NOCAM) path.

Pins absolute values — an output digest and the RNG operation sequence. The
sibling suite's ``test_default_path_byte_identical_to_explicit_defaults``
compares two legs of the *same* checkout, so it cannot detect a change that
moves both legs together; the frozen manager hashes cover manager cfgs and
scene scalars, not the generator's geometry, defaults or RNG consumption.

Golden literals are the generator's output before any enrichment argument
existed. A diff is a contract break, not a test to update.

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
# Stub scene (mirrors the sibling enrichment suite's harness)
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
    "s20260101-d7-b1": "c05d1858900147e045f1d0bf61d78dd7cc904bfb75d861b0f158190b3b688f15",
    "s20268020-d0-b1": "3db0d9f94dd8a4b1423748b323cfe416e32d0141297ff917c43788c85f93dc77",
    "s20268020-d1-b1": "d3538669802fbdc547f43fbc3a7983485a45322bbf5ebc5691026004da233568",
    "s20268020-d2-b1": "e9e6aae3d18787c31c70bb134411d7ac948acbf9f6e59c61b69c44fcb83aa16c",
    "s20268020-d3-b1": "17c8d9386cbc02ece65fafecb64616ce6ee80385a25db1ac1b0a3519891c25cd",
    "s20268020-d4-b1": "ded6ba886dbc646b5a50d8ef5067005d85737c32922ff13b9dbb52fe530363b8",
    "s20268020-d5-b1": "948c3e5d4c1705115c613584eb5445db68b44eb61361ed7381b83eeb4d9cce75",
    "s20268020-d6-b1": "d68637819efa7c0917c5351195201a27f0339e9d463eb2127508a00cf54074a7",
    "s20268020-d7-b1": "05551168f503b309f9d6903b5bce0fa2bf11f209025038bb482bf05ad50454e9",
    "s20275939-d0-b1": "a98f6d44eb544e683264f080d55517396a46d5f18162440dd663de09c212c001",
    "s20275939-d1-b1": "749cc681597ab38026e15a56416d563157c7b71543ca906a6542a21492a80d9a",
    "s20275939-d2-b1": "a57f6efe4c9fa3b55c70b73c2c343594151f588496831094a10363bcfa5fa9a9",
    "s20275939-d3-b1": "a075ade0c1ed2495131b6b236eea16357792385178e52425a7c493fc9050c4bd",
    "s20275939-d4-b1": "a922ac97b3e36e9f2b4c903461de071629e1c110768c7f7eaaf9a7b036f95e9a",
    "s20275939-d5-b1": "cebf2d5cafc54503b09e22ccc0036a1842f3d436e761b3784d147d495bd5d150",
    "s20275939-d6-b1": "26342faf5eeb39988ac3d047d642a04d03d16fe467ae375bc52c0d009942aefe",
    "s20275939-d7-b1": "498fc945a62fdda722211ddd51906056263093039d7a4db26f0cbc560840f935",
    "s20283858-d0-b1": "39fa5390be2bbab67ff999f7c2c1a09fd15086e629d3ea9186fb664ac05fa699",
    "s20283858-d1-b1": "171fd1d4a248d68fd58e8f1362194d7074f07553bfeec89953878093e44357ba",
    "s20283858-d2-b1": "b11d41a81598c425ec518d875963f67040ea17f5fad86d46813e5a5b53f403fd",
    "s20283858-d3-b1": "3aa177ff7c8da3f470e2e7c6ad7bed905bbb9a8ecd6a8534ffbff185a27e809d",
    "s20283858-d4-b1": "aa2c451aa97285446ef889815a07b9fb36eebd600440a1005835222ce1c7ba48",
    "s20283858-d5-b1": "b191483c4e2cbb226a4f47e4f8b4630037a49ed1ef8f2b79037600a18e5c5750",
    "s20283858-d6-b1": "340aed607c99b63b57ce82ec63d3022d98c94cbb36fb356d320dc643042f7224",
    "s20283858-d7-b1": "8c3c0142fa1606f9fc4f8bd293ce3bcc053719bb10b9375369babb855aae2eaa",
    "s20291777-d0-b1": "6e6f341b7a86cdc7e0351f369789ee61ee9ddfc1bb852839690d9c1b1d0560bf",
    "s20291777-d1-b1": "9e747b59d4bdb978b0d0e68987df26f08d9b91efa5d4dab1ce47a2edc4acd88d",
    "s20291777-d2-b1": "e91e7bb2b831a0b5c37b8435d6657120fc53273aa81894e77707ec55b94c4d6e",
    "s20291777-d3-b1": "71afd75fb5e8d585b2d98815aca037503c275d094a54ea0f5927a876a0f0d9e2",
    "s20291777-d4-b1": "97d95a55202c5843775f480e5f2b3bbcd500bde6990ef551154dd8aa7b499fbc",
    "s20291777-d5-b1": "35d6674e7c684aef0c2f426ecf3726b9b98aba357411cabdd6929f64fd499d6a",
    "s20291777-d6-b1": "b3861579626512c6961e4e9d64c16a393052df489bcf67b857fec3ae12968c67",
    "s20291777-d7-b1": "0c998ce635feda85009719158381ebcc96571113f17e5cbb96150499627464f5",
    "s20299696-d0-b1": "6d608bd608b2cd20c4d93ab0bc151df0cc34bd8455acfb7dcae8f07e5a8ef02a",
    "s20299696-d1-b1": "94fbea5563227ef9a57eef476cc5cc6fbb6136d0b97995cc121cdfbacf00096b",
    "s20299696-d2-b1": "6e42e84b0d9e853b0fca1a7366765dc7886c7ef72ae0adfed1fb79a311c534fd",
    "s20299696-d3-b1": "60eda8d30be3df8497f6f338202b4443ca43c2972cd6d9df4beb8dd09935c1eb",
    "s20299696-d4-b1": "1d916eef19983c0cd8d23dff33ec946c65679fb65b78510760434b7328b4b29c",
    "s20299696-d5-b1": "27e638cf28f41a4c23b233174224b33ffe6c188d86116d6078dfa8ecfb646bd8",
    "s20299696-d6-b1": "3814f3d26980b3094d08a929ed94a0ae0d1f13f0a1ade81540bc7492476e425b",
    "s20299696-d7-b1": "9a057dd945c56f8cef9fc80149947d3ace7fe21ba7fd8f55c3f6ddebf2f4b201",
    "s20307615-d0-b1": "83ccf36a7322386540bc7acd8a5decb65d3ef862f35ec3f4f4fb17d74385cba7",
    "s20307615-d1-b1": "53a3a4769f66eb98af822a147235ab1c7575f93b934b73e3082525d03b09ef07",
    "s20307615-d2-b1": "47602d2a0a0435eaecf8efd647f7a4e6e8dad69aabd970bb2e8a2ee72aa1579b",
    "s20307615-d3-b1": "a6a4284814269942164b51c387e120265b192f36e1938dad3ddae52e594606e7",
    "s20307615-d4-b1": "982754bd6e45728dab9fae3110268aacc26358ea81fb9ba043243576817d0453",
    "s20307615-d5-b1": "29f4b8648e5d4a70003627d17360ebd1c8eeeb02321afd69361bb16363f75389",
    "s20307615-d6-b1": "e377983f8d3623bb90297112eda2c1789bf392265658cef499cfb297fdd9ba95",
    "s20307615-d7-b1": "979589be02032b4619da7835661ad4cbfaa826a7c4ab918d47382590582630f4",
    "s20315534-d0-b1": "1baf901c3ce927a69692a86f133d83e8a724375e7ee2f079d1e5f5b0b7b900e0",
    "s20315534-d1-b1": "6bf701efc567ba47378939f4035b2237874f5dc42913d189ba7f501c9bf952c7",
    "s20315534-d2-b1": "a7431fbf7a9748e4e6601bbc22f9899b80c73a0dbf681234061d959c9977318f",
    "s20315534-d3-b1": "bb97e0bbe675f68859e4f7a31cb68bb2adf7c729d11c13ac04af7d7557aa269c",
    "s20315534-d4-b1": "86488c8db2a8f54fda144aa0951556ee138aed26df1f582d395d44735aae5d06",
    "s20315534-d5-b1": "9db0ada56ab1203815794aae54a23bdce748978d61da04fa19a37f6b630aca3c",
    "s20315534-d6-b1": "f930bae49a0c5d6d9291253c7208a8f1e6644ea6234ac27bbe8a9c695c39f3b4",
    "s20315534-d7-b1": "0a38655382ae07f738281b7c4b11fa254238f88dae2988b0fa8dc354cf57ce89",
    "s20323453-d0-b1": "0c04439764ef5c7e20d83dff06753701f41118bff943fb19dc0a55790ebbba35",
    "s20323453-d1-b1": "3ef6f8af69a6b6089529e9bac8734c7527c5f118e43ca0655ff558c120666e39",
    "s20323453-d2-b1": "e8725aa9ca0689a1167b3184b61ec4d549e12fa3eacb4f1dfb324744413ca96d",
    "s20323453-d3-b1": "6850a4649cb9ff995de488d3757ffadec147640cd01ff1071375e674d3f3eac3",
    "s20323453-d4-b1": "71a1a6f0fa58c07aa0199dc3e3705f3d7a6503e19b5992ff6cb0f92a1f5a715f",
    "s20323453-d5-b1": "726bb8d7d4bf9d36c6e123a4c93c07b7076294facd53ac7ce43ab2e7f00410d1",
    "s20323453-d6-b1": "041a0116f6c0bd2444559b438897e498b3bf38d926fca5a4f8c38d656200c347",
    "s20323453-d7-b1": "3e2373719fe6b5e80a63eb9a45c1349258c4a15f7af89d4c783551884ab07f04",
    "s20331372-d0-b1": "0335d2ed5ab321d90a7f1ea74525f6c69a2bbfc421787472b21126f4eaf8f5c5",
    "s20331372-d1-b1": "c3f9368943bffd884d1b4551e2641158cd6fc621a4e494bd309b9c13ccf99a51",
    "s20331372-d2-b1": "250600d9a8214beb68528f655c9639dcd59cb0d25972ba27f5bead2475b28460",
    "s20331372-d3-b1": "d3a9187a584a53d4cbad11e8f9f6dc5fe50b4f75267cc242c46c8e4e71776779",
    "s20331372-d4-b1": "52ef2bb779d50232f996a22fc8609c493366cb5bb0de52f315c337af91ff1d24",
    "s20331372-d5-b1": "4c5bf0f755c8dd4aa3cdb23c66f3c5bd55b0fb80e7f4771c85530ef9ff961ce9",
    "s20331372-d6-b1": "a7a41caef14875cc3e9bcf27632181873ffed8b67840569818bab55fbd8ef79a",
    "s20331372-d7-b1": "e625ce2e3258ecf729ce2012aebe80fb2f7214675a35ca9a6fae18f70c847ebc",
    "s20339291-d0-b1": "632f6eae0bfa660184389df94cc289ad6f8983497e8fbc502df035c72781e9fb",
    "s20339291-d1-b1": "7559d36c1423e4fbf86e1558295b360d8d4d4e546f88f4a93bc1d6864ffbb48a",
    "s20339291-d2-b1": "1e1dcce39f19862b829df5712990fdab927657a8277d9ac4194d5b6ba39250f2",
    "s20339291-d3-b1": "63ccb96e2689553200bed46c931a44a429fff6b7557a55c97dc622d725082787",
    "s20339291-d4-b1": "0dd9cfec8d0a6cc5772039a28211790ed42c23d99880e46b17f1d4a6060a52de",
    "s20339291-d5-b1": "9edaf35e546d563e552699d76dce90f692e909adca423485122ab4369ed9a200",
    "s20339291-d6-b1": "76c311110133053716c7547197a56d2b52c4de90ad45c197129a0506692b07f3",
    "s20339291-d7-b1": "6699f8b256df9ceb6073910db2b704325fee986dbb4ea1ecb8428261c7915740",
    "s20347210-d0-b1": "82cc66e9d989d728ddf31020ca01dad07ce02c0c272c00e449352ddae0234241",
    "s20347210-d1-b1": "36857015196968a08fb9591be73240ee6b00fd24e10f65c490dc1c4418b503dc",
    "s20347210-d2-b1": "367042b8c92480e280f7c857f9b1346187568d30f6847797f88cb64910f9a4f1",
    "s20347210-d3-b1": "276bb306fa23d529cfa4a454e1df053341ac5d65750627d1ac2b2e3c716a89f1",
    "s20347210-d4-b1": "113be08061d840bf35b39a3f6501df713109498121f9d873db23ee6b5dc41334",
    "s20347210-d5-b1": "2a7201bd58d2a80687dfd76df56f31c57e986d233f703e80e169457f9870a2fe",
    "s20347210-d6-b1": "3f0925ffcd6fcfdc2841c919b10c45c1ed1f3d5fe8ec9bf4046818eb780b67a5",
    "s20347210-d7-b1": "7688f8c8e3110f13ac36dc381c82b51ea0fa0e8df305886e878e837d47e25ac1",
    "s20260101-d0-b8": "ac683efe2de4d8d6d3d2b10fb100508e2d215c619cb723f19fcf51618e837aaf",
    "s20260101-d1-b8": "4416cd661d30c177bec26551d7545fe92717a4f82caf9746a05fbd0fe746ea6b",
    "s20260101-d2-b8": "91629c7bfd9687924cf10e2247e1cd897c3b376ab8f1d68056499d55577ade63",
    "s20260101-d3-b8": "4e90113a51491524dbd7d08baaa4c47a1cb840ae42c63f78acfa4a8a46c934cd",
    "s20260101-d4-b8": "cbbba3b845ab7fdfbef66df5dc23fdedb9b33795544aa97513cf1d857318c3d6",
    "s20260101-d5-b8": "d4636524b7a3919203a3dde84f55811dd33d3c1136f6e48e9d6b7aaf6dbaa373",
    "s20260101-d6-b8": "91e3c2453fc9c15540dd0337f44cba9b491927afeee1cc65921b200396cdaa31",
    "s20260101-d7-b8": "cf8fa51080be7a44dc8f120108af043d7508ad52016e29f98c5d9278d5bccf3a",
    "s20268020-d0-b8": "a194d0bf054824534e2f056cdd70f5ca912faa6c740764f864e41b7f5040cfce",
    "s20268020-d1-b8": "ea262b536f201c4ff217bd395e4073ad56a4439c4d7d15b7b85de422f37e377c",
    "s20268020-d2-b8": "912c4d9a19dffb560a7be1e70e1c39094d36e59622c71e9de7b6c4edb1446f96",
    "s20268020-d3-b8": "80b760c7afd9c8344fdb31c1da260e99d8af1a059479167894216886819f26ec",
    "s20268020-d4-b8": "70ce91e1d35068c9f30af64140a3eb5812451aa5d6e2f7e67ceb1bafb4c3a535",
    "s20268020-d5-b8": "79359e0ac663a012ce3b406a45b7c16710438b78a87a8ba27562688cfa84f52c",
    "s20268020-d6-b8": "0ae2a5a6928323ac6f6b680585e8d8abceef56fe1a565c6e17611b75faf1b0d2",
    "s20268020-d7-b8": "756462f31caac6876315f6114af2bbcced1ac8160651e64a628828c8bc760704",
    "s20275939-d0-b8": "69ed4c20a42b54e1bf297fc21a256a08fead26fed7f27c4b69acc96eff5eb418",
    "s20275939-d1-b8": "c72ea73a5c7d848fc80bab8a67cd10309a625b9d7e9fc91f903c7f534973a3a3",
    "s20275939-d2-b8": "74a01c3aff6ce5b7b1c2efd0f2ad9db02aa30cbbdddccfbcf35eb86e5d3cefef",
    "s20275939-d3-b8": "32e26526c8a42d501508f3bbb607ebb72d9c844112f26a69271b1ac802a6e7f1",
    "s20275939-d4-b8": "629b73ba696fc01648b220e698542acc23985a0869dd72814344675e9cce7b7e",
    "s20275939-d5-b8": "4ccbab566262b69b48b9124e874e8636a523b28d7bf2fb55a73982db0fd18e82",
    "s20275939-d6-b8": "846d2608dbb53cb2d9f035e396991d36554ba4a4d83b554694146ba04eb698e4",
    "s20275939-d7-b8": "3e05ae68fc6b0e649ef43df3c1793030296a11336b834a026ee7f9ffc01b6203",
    "s20283858-d0-b8": "9dddff61c94a98a91ea7b9ffcba0f72fa13c0dac34127a169298c0e0c3902839",
    "s20283858-d1-b8": "a8ee8e7c0e8863ad8b230f9cc99c1d987d03e31855e8e2d4b28247ed9f4fed5e",
    "s20283858-d2-b8": "ead4808d29f5a659ccdb6ea5fa83fe45c1e1be1152f8cf13bce67e9db2063fd9",
    "s20283858-d3-b8": "896b74208813265629f4a6ab643f17a3610949fb732193162a39112a7b59211a",
    "s20283858-d4-b8": "f1f1bef76b4e1706a87adec691b1c4a07262843e4bcdff9f9002e3ce5e821060",
    "s20283858-d5-b8": "d65a8a1b0e0ae773290645cab6263abc9c9e60ac393450739fdf5395eb3b4a68",
    "s20283858-d6-b8": "025e2a807e09cafa305697c5465db463b2a0dd9723ceba03b50743715cce21db",
    "s20283858-d7-b8": "1c1c554d58ab77ecd3451d00c32618ad12818bc891e568417341024974c02668",
    "s20291777-d0-b8": "6d4bdcbb67650433d16b47fb3ad4f6d67ad6e9894dfd416551b0dff2a9215bac",
    "s20291777-d1-b8": "b95d3f6f638157e125a51f90bb8910dbd30095405d9d3e83d4962f97de9a0ae8",
    "s20291777-d2-b8": "ceb9e998bd426345943523877ecf478b396d016530db620fc540e6b4c106e03e",
    "s20291777-d3-b8": "0645b5f0a7daf10aadcfb6c83f1967760f41500f6dd6c19afcd90945e14fa9a2",
    "s20291777-d4-b8": "dc325cf57c5f1586731c831e718a642e7d02cd4651261e8ada149b8cddc1217d",
    "s20291777-d5-b8": "57a805abe24adabaff262a4cb7f779c92fefeb2a671d150186ad47b66afb13d8",
    "s20291777-d6-b8": "dd28d56f3ae0ad9f8d780db42e4e2b07da209661516c0007d14c7b7799d7611d",
    "s20291777-d7-b8": "766238be283cbfa27ad2ab8751605faf8fdbed677b88ae835579832b214062a7",
    "s20299696-d0-b8": "c1854308eb940972851a064d409f3f744b8e3c18c7c9b39d175c3192379c2c3e",
    "s20299696-d1-b8": "d81fd8ff09854b76f0717782d9aaf822d9edff8f7a7b75f61e50302313dbfc6d",
    "s20299696-d2-b8": "20ff1756ef985f72fc6d19584b39a1827b2427d6304a82105b4aa7bf7d0f0efc",
    "s20299696-d3-b8": "53012338bbfd1fdfa0127097b41907a1e5a88c8747ab1535a40088eadadf1a19",
    "s20299696-d4-b8": "5ca287d326a0e93547190982ec4bc207d3221839070d5ff05346a728ad8529d8",
    "s20299696-d5-b8": "32d2d594da707fe18a0b2180796e321b35c0c9a90adb9488e01521c6d260ddcc",
    "s20299696-d6-b8": "9d1faea148daa162f1c66af37217cb4db55ffd3792751a8f2f76bd3573c00257",
    "s20299696-d7-b8": "87b87a9ac5facf4f9194133edc4308d00b85a6d7fd9822d56d474d69d2284fc4",
    "s20307615-d0-b8": "31d77107698062b956f35b658162297547f4f971f2ebb1dd95e6e4c00899a1b7",
    "s20307615-d1-b8": "74f2eac85e11534cc9db9c071ba3ba4d80fb8952568570ef70164213a49dd002",
    "s20307615-d2-b8": "d27d6576e98337483c2f92959652feeab65f4be952630b101bb3d58c6735e578",
    "s20307615-d3-b8": "a55c4c77f1b9d6fb21c724af5379c52c482b8f25743b6282fcbca1c3c41e481a",
    "s20307615-d4-b8": "fb689c554871ab0e1c570f6f05249d96aba3bdf62082b39f3b81825883040023",
    "s20307615-d5-b8": "486b97236acd3ee47ceaa6bb462e9dfc48e44dbe057f3e3620391addcdd13b8b",
    "s20307615-d6-b8": "614d0032ae6c6a280c73d836f10e2b9f1fc0dc305c3ab58a90677aaa3c55c6e2",
    "s20307615-d7-b8": "f02424a61e318eaa3e5df88e960ad9265385638f634e5fdceadc8c95db336710",
    "s20315534-d0-b8": "3097e1b0847e22f5b628331b92cae09efdc99dc7d9be425169fd372fae065050",
    "s20315534-d1-b8": "e83a7a24b09628377acc9b81119c5ab78c503254714363f1ce5c14e53dd35319",
    "s20315534-d2-b8": "757c3ffce17bebd119b3383c167e8b2c261eee22acd28d4d8e6087a8c4c98a32",
    "s20315534-d3-b8": "113943bc3e503730b40206f277d6b5dff5007544cfcdfd76c37f72a4bf5f1518",
    "s20315534-d4-b8": "b15cbb8efb25d74ff3c09f8485db702133ce0cfb8d26ac83643531720372351e",
    "s20315534-d5-b8": "73e99055aa24bf3e932c31acdef36fb104310611a3ac04bd1495bb6433bd8c94",
    "s20315534-d6-b8": "6fac0ff34a257a34287f3295e2cac449c41dbbf9832b16a72e1a343ace735871",
    "s20315534-d7-b8": "560437be1f4371ba960bf2bdaf65e2fa0a87f0d81b26e03e62f84ff52174ad07",
    "s20323453-d0-b8": "9c7ae8c2877251d6cd2bae8b19062c224e0f883ac5c42ec57b44e12ed9c298cf",
    "s20323453-d1-b8": "090093364207adf428af32b276697e93a92bd2ce4a1769994dcb3871416b42f9",
    "s20323453-d2-b8": "b88f3b4436b4f0c097b5f815d6c6ebe0a7b480b8125ca6d921d1e7e4621bbb38",
    "s20323453-d3-b8": "632290fe03bfdd29f8fd28654a12432ecdc1af1a8cad2af17067ddd11c7a2e8c",
    "s20323453-d4-b8": "4815cbca59be73b572bf9ea68e0f3fe2ee1d66d3c24df1ad6070a9acfff32fca",
    "s20323453-d5-b8": "69c1a0630fbdff5c8c6f7381dd8af1167f94309a7eae42dd2727a4dbac8b9bbf",
    "s20323453-d6-b8": "1724386dd84f3c83359392cf6b88a2f34b682db69aa5b7f699fe0fd5f6772d0f",
    "s20323453-d7-b8": "506a28682766b131f5669d2c55f361239f78ff54482e9843f8826d65f473e83e",
    "s20331372-d0-b8": "85cec68a9b7c467e7b9744764080d83ebf02299498fe22be743331ee6e3b69b5",
    "s20331372-d1-b8": "a6cbd967e84d69232b0af20f08c1473fc85bce787dfede50576aeb28dfc40141",
    "s20331372-d2-b8": "f822157acd7a173e752aac2da6da2b7b808973331098b90da09a6a8f02abddfe",
    "s20331372-d3-b8": "313693f39742972cb0f608fec04a08ff5d13d31d2c0a8fe31bbf1d0e271088de",
    "s20331372-d4-b8": "2718594f0f75cff3f31bc87fffbd0829978fec67b93083397926d92353aaeeab",
    "s20331372-d5-b8": "e07431adff6b6f9def06ddb79106ef388325aabb17914b06862f9ddebcb1c994",
    "s20331372-d6-b8": "a7502f115922e6d635387e8838e4f3955cda55304cb8298612aea5a0f3901fce",
    "s20331372-d7-b8": "04636d4dd37676f5d24ec84057c6613f50b4298f1e4fd96126432ff82cbb2260",
    "s20339291-d0-b8": "9fd9d0bb5c46ba426ffc0537af2b0e1eb6adc97c1a2418756ec3ea7ba7c21d11",
    "s20339291-d1-b8": "2fa75cec9393069060277a8a9013d48b123a62cf8de06d52f78659d69a07467e",
    "s20339291-d2-b8": "0e437a2cba1e22336b9f4a5eac1775877a2f8519843eba5b9bea8f165484cf1f",
    "s20339291-d3-b8": "d9554ca7bf1d09476a1342d4f3c1a00822e2b439236657e49bc0555f9a66aada",
    "s20339291-d4-b8": "8c45898f99160cad79b93c08fd7a47352e37ba4091d8e80911f940a61688a2e3",
    "s20339291-d5-b8": "19b823ec3d845794ef5a65f90b4accc5f47fb283cc3bda89f9e585cdb4d1d4bc",
    "s20339291-d6-b8": "ddb4a5614c590d87d4485d5ba6201d771d08ddfc041c81743efead6cafab5d6d",
    "s20339291-d7-b8": "6993d0c7ef70b916bd1ffc4ef30ec15263cb4cae316a5ea4fc0158f85fb5e1a1",
    "s20347210-d0-b8": "75a66e64903bd3cbfcd572b36d06f4868b76ceebf33fd87a49cdfb7e250ea0e7",
    "s20347210-d1-b8": "1b8d61f6a6c05efb4a3134de0231622921889b2457a2e1aae2f633f2762e425b",
    "s20347210-d2-b8": "4cf80d9babf05be25b361ece653bce926c4f9ef07ceef7e19dde2aa7f49360a1",
    "s20347210-d3-b8": "f06e8868fd8af71159f31755b82874a80a1e28175ff79279bd6a362f468b970d",
    "s20347210-d4-b8": "c638c55ab063b757ec9cff1f85be2319f28f7c4d3ae99859d35dd6ba3f5d5d96",
    "s20347210-d5-b8": "e47df05afd4d943e451c1cc31cfe03eba2f22269a2723c260272b273043bc2e1",
    "s20347210-d6-b8": "edbe64d20816f33500d4df86ef5477c7313a0ce2277c0b4c9285ed511e542d8e",
    "s20347210-d7-b8": "1d2529df8cab0b131852cf0f99b91b318fc23174a53808cb683a176660730cd8",
    "s20260101-d0-b64": "ca1065bd1dce4cdde23cb8efda128a86858b6167cdc8376779278eb2100ac795",
    "s20260101-d4-b64": "c009e2f153ab456c990acd84b717be216f757c3518afd36f75fe1c82a8f02f33",
    "s20260101-d7-b64": "dd22a8ea6d48fffd4f1dc6c36b69cc79f300ae37f0120e77daa75e651828ee3a",
    "s20268020-d0-b64": "2cbe46ae73a57f18e48d316271275e2c6c8d1d174c53783c98a4866e349e04cf",
    "s20268020-d4-b64": "69c816642f925e3d65a90d720869bb434ebd73d9ac826aec27c1e34709eeb45a",
    "s20268020-d7-b64": "1fe86d43e29137818ce2b67059c8cf5a239fded4e664825c7748ec4769aa9efb",
    "s20275939-d0-b64": "7462f83464493ef3b90ad4448841de45b13b21750ec77345903fb71b1757f5de",
    "s20275939-d4-b64": "bc551f38b8e1dd1eb658badce4ca35cc7e01e79a8f17883c6dd87c70afa7dc24",
    "s20275939-d7-b64": "4c849fe9a2b6cfd98e524f8cec8b97e29286d510cdb7cc08086d4876ef15a0f1",
}


def test_default_path_output_matches_pre_enrichment_goldens():
    got = sweep_digests()
    assert set(got) == set(_SWEEP_GOLDEN), "sweep shape changed — goldens are stale"
    diverged = sorted(k for k in got if got[k] != _SWEEP_GOLDEN[k])
    assert not diverged, (
        f"{len(diverged)} of {len(got)} default-path cases diverged from the frozen "
        f"pre-enrichment output; first: {diverged[:5]}"
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
