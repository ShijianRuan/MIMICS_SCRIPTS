"""标注版本回退链的单元测试。

覆盖 Action1_ConvertLabeledToTrainData 的版本管理纯函数：
- _version_dirs / _scan_minor_versions: 大版本/小版本回退链, 小版本按序叠加
- _resolve_organ_path: 逐器官取回退链中第一个存在文件, record 回调
- _resolve_subject_version: 多数据集时按 subject 源数据集定位版本

这些函数是纯逻辑函数（只用 pathlib/re），不依赖 pandas/nibabel/scipy。
为避免 Action1 模块顶层重依赖在轻量测试环境中不可用，这里用 importlib 加载源文件，
并 stub 掉缺失的第三方依赖，仅取出目标函数。
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _stub_module(name):
    """生成一个 tolerant stub 模块：任意属性访问都返回可调用占位。"""
    class _Tolerant(types.ModuleType):
        def __getattr__(self, item):
            return _Any()

    class _Any:
        def __call__(self, *args, **kwargs):
            return _Any()

        def __getattr__(self, item):
            return _Any()

    return _Tolerant(name)


def _load_act1_pure_functions():
    """加载 Action1 源文件, stub 缺失的第三方依赖, 取出目标纯函数。"""
    for name in ("pandas", "nibabel", "scipy", "scipy.ndimage",
                 "nibabel.orientations", "tqdm", "ResampleImageAndMask"):
        if name not in sys.modules:
            sys.modules[name] = _stub_module(name)

    spec = importlib.util.spec_from_file_location(
        "_act1_for_test", SCRIPT_DIR / "Action1_ConvertLabeledToTrainData.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_framework_pure_functions():
    """加载 AutoSegmentationFramework 取出 _normalize_annotation_version（纯函数）。

    stub 掉 SetEnvionmentVariables 本地依赖避免连锁导入。
    """
    if "SetEnvionmentVariables" not in sys.modules:
        sys.modules["SetEnvionmentVariables"] = _stub_module("SetEnvionmentVariables")
    spec = importlib.util.spec_from_file_location(
        "_framework_for_test", SCRIPT_DIR / "AutoSegmentationFramework.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


act1 = _load_act1_pure_functions()
framework = _load_framework_pure_functions()


def _names(paths):
    return [p.name for p in paths]


class TestVersionDirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.subject = Path(self.tmp.name) / "s0001"
        self.subject.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _dir(self, name):
        d = self.subject / name
        d.mkdir(exist_ok=True)
        return d

    def test_empty_and_v1_only_base(self):
        """None/空/"v1" → 仅 segmentations/。"""
        for ver in (None, "", "  ", "v1", "V1"):
            chain = act1._version_dirs(self.subject, ver)
            self.assertEqual(_names(chain), ["segmentations"], msg=f"ver={ver!r}")

    def test_major_version_chain_no_minor(self):
        """大版本 v3 → [v3, v2, segmentations]，不叠加任何小版本。"""
        self._dir("segmentations_v2.1")  # 即便存在, 大版本也不取
        chain = act1._version_dirs(self.subject, "v3")
        self.assertEqual(_names(chain), ["segmentations_v3", "segmentations_v2", "segmentations"])

    def test_major_v2_excludes_minors(self):
        """指定大版本 v2 → [v2, segmentations]，不含 v2.1/v2.3。"""
        self._dir("segmentations_v2.1")
        self._dir("segmentations_v2.3")
        chain = act1._version_dirs(self.subject, "v2")
        self.assertEqual(_names(chain), ["segmentations_v2", "segmentations"])

    def test_minor_accumulate_chain(self):
        """小版本叠加: v2.3 → [v2.3, v2.1, v2, segmentations]（minor<=3 降序）。"""
        self._dir("segmentations_v2.1")
        self._dir("segmentations_v2.3")
        chain = act1._version_dirs(self.subject, "v2.3")
        self.assertEqual(_names(chain),
                         ["segmentations_v2.3", "segmentations_v2.1", "segmentations_v2", "segmentations"])

    def test_minor_only_accumulates_le_min(self):
        """指定 v2.3 时不取 v2.5（minor>3 不叠加）。"""
        self._dir("segmentations_v2.1")
        self._dir("segmentations_v2.3")
        self._dir("segmentations_v2.5")
        chain = act1._version_dirs(self.subject, "v2.3")
        self.assertIn(self.subject / "segmentations_v2.3", chain)
        self.assertIn(self.subject / "segmentations_v2.1", chain)
        self.assertNotIn(self.subject / "segmentations_v2.5", chain)

    def test_minor_accumulate_picks_latest_for_conflict(self):
        """两小版本都改同一器官时, 链中 minor 大的更靠前（最新修订优先）。"""
        self._dir("segmentations_v2.1")
        self._dir("segmentations_v2.3")
        chain = act1._version_dirs(self.subject, "v2.3")
        # v2.3 在 v2.1 之前 → 同器官冲突时先命中 v2.3
        self.assertLess(chain.index(self.subject / "segmentations_v2.3"),
                        chain.index(self.subject / "segmentations_v2.1"))

    def test_minor_excludes_higher_major(self):
        """指定 v2.1 时链不含 v3（不向更高大版本回退）。"""
        self._dir("segmentations_v2.1")
        self._dir("segmentations_v3")
        chain = act1._version_dirs(self.subject, "v2.1")
        self.assertNotIn(self.subject / "segmentations_v3", chain)
        self.assertEqual(_names(chain), ["segmentations_v2.1", "segmentations_v2", "segmentations"])

    def test_cross_major_minor_no_other_major_minors(self):
        """指定 v3.1 → 链含 v3.1, v3, v2, v1，不含 v2.* 小版本（跨大版本不取小版本）。"""
        self._dir("segmentations_v3.1")
        self._dir("segmentations_v2.1")  # v2 的小版本, 不应进 v3.1 的链
        chain = act1._version_dirs(self.subject, "v3.1")
        self.assertEqual(_names(chain),
                         ["segmentations_v3.1", "segmentations_v3", "segmentations_v2", "segmentations"])

    def test_minor_on_v1(self):
        """v1.2 → [v1.2, segmentations]（v1 是最低大版本）。"""
        self._dir("segmentations_v1.2")
        chain = act1._version_dirs(self.subject, "v1.2")
        self.assertEqual(_names(chain), ["segmentations_v1.2", "segmentations"])

    def test_minor_without_other_minors(self):
        """指定 v2.5 但磁盘上只有 v2.5 → [v2.5, v2, segmentations]。"""
        self._dir("segmentations_v2.5")
        chain = act1._version_dirs(self.subject, "v2.5")
        self.assertEqual(_names(chain), ["segmentations_v2.5", "segmentations_v2", "segmentations"])

    def test_invalid_version_raises(self):
        """非法版本号应抛 ValueError。"""
        for bad in ("v", "v0", "v2.0", "v-1", "abc", "v2.x"):
            with self.assertRaises(ValueError, msg=f"ver={bad!r}"):
                act1._version_dirs(self.subject, bad)


class TestResolveOrganPath(unittest.TestCase):
    """问题5核心端到端: v2.3 改 liver, v2.1 改 spleen → 指定 v2.3 时 spleen 来自 v2.1, liver 来自 v2.3。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.subject = Path(self.tmp.name) / "s0001"
        self.subject.mkdir()
        # 基础版: liver + kidney_right
        base = self._dir("segmentations")
        self._touch(base, "liver.nii.gz")
        self._touch(base, "kidney_right.nii.gz")
        # v2 大版本(全量)
        v2 = self._dir("segmentations_v2")
        self._touch(v2, "liver.nii.gz")
        # v2.1 小版本: 仅 spleen 增量
        v21 = self._dir("segmentations_v2.1")
        self._touch(v21, "spleen.nii.gz")
        # v2.3 小版本: liver 修订
        v23 = self._dir("segmentations_v2.3")
        self._touch(v23, "liver.nii.gz")

    def tearDown(self):
        self.tmp.cleanup()

    def _dir(self, name):
        d = self.subject / name
        d.mkdir(exist_ok=True)
        return d

    def _touch(self, d, fname):
        (d / fname).touch()

    def test_base_only_when_no_version(self):
        chain = act1._version_dirs(self.subject, None)
        liver = act1._resolve_organ_path(self.subject, "liver", chain)
        kidney = act1._resolve_organ_path(self.subject, "kidney_right", chain)
        self.assertEqual(liver.parent.name, "segmentations")
        self.assertEqual(kidney.parent.name, "segmentations")

    def test_major_v2_uses_v2_then_base(self):
        chain = act1._version_dirs(self.subject, "v2")
        liver = act1._resolve_organ_path(self.subject, "liver", chain)
        kidney = act1._resolve_organ_path(self.subject, "kidney_right", chain)
        self.assertEqual(liver.parent.name, "segmentations_v2")
        self.assertEqual(kidney.parent.name, "segmentations")

    def test_minor_v23_accumulates_spleen_from_v21(self):
        """问题5核心: 指定 v2.3 → spleen 取自 v2.1, liver 取自 v2.3, kidney 回退 v1。"""
        chain = act1._version_dirs(self.subject, "v2.3")
        spleen = act1._resolve_organ_path(self.subject, "spleen", chain)
        liver = act1._resolve_organ_path(self.subject, "liver", chain)
        kidney = act1._resolve_organ_path(self.subject, "kidney_right", chain)
        self.assertEqual(spleen.parent.name, "segmentations_v2.1")
        self.assertEqual(liver.parent.name, "segmentations_v2.3")
        self.assertEqual(kidney.parent.name, "segmentations")

    def test_minor_v21_does_not_pick_v23(self):
        """指定 v2.1 → 不取 v2.3 的 liver（minor>1 不叠加），liver 回退到 v2。"""
        chain = act1._version_dirs(self.subject, "v2.1")
        spleen = act1._resolve_organ_path(self.subject, "spleen", chain)
        liver = act1._resolve_organ_path(self.subject, "liver", chain)
        self.assertEqual(spleen.parent.name, "segmentations_v2.1")
        self.assertEqual(liver.parent.name, "segmentations_v2")

    def test_record_callback(self):
        """record 参数正确记录每个器官来源版本。"""
        chain = act1._version_dirs(self.subject, "v2.3")
        record = []
        for roi in ("spleen", "liver", "kidney_right", "nonexistent"):
            act1._resolve_organ_path(self.subject, roi, chain, record=record)
        record_dict = dict(record)
        self.assertEqual(record_dict["spleen"], "segmentations_v2.1")
        self.assertEqual(record_dict["liver"], "segmentations_v2.3")
        self.assertEqual(record_dict["kidney_right"], "segmentations")
        self.assertIsNone(record_dict["nonexistent"])

    def test_missing_organ_returns_base_path(self):
        chain = act1._version_dirs(self.subject, "v2")
        missing = act1._resolve_organ_path(self.subject, "nonexistent_organ", chain)
        self.assertEqual(missing.parent.name, "segmentations")
        self.assertFalse(missing.exists())


class TestResolveSubjectVersion(unittest.TestCase):
    """问题4: 多数据集时按 subject 的源数据集目录名定位版本。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # 两个源数据集
        self.ds_a = self.root / "Totalsegmentator_v201"
        self.ds_b = self.root / "TotalsegmentatorMRI_v200"
        self.ds_a.mkdir()
        self.ds_b.mkdir()
        self.dataset_path = [str(self.ds_a), str(self.ds_b)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_value_broadcast(self):
        """单值 → 所有源共用。"""
        subject = self.ds_a / "s0001"
        self.assertEqual(act1._resolve_subject_version(subject, self.dataset_path, "v2.1"), "v2.1")
        subject_b = self.ds_b / "s0002"
        self.assertEqual(act1._resolve_subject_version(subject_b, self.dataset_path, "v2.1"), "v2.1")

    def test_list_locates_by_parent_name(self):
        """列表 → 按 parent.name 定位源数据集取对应版本。"""
        ann_versions = ["v2.1", "v3"]
        subject_a = self.ds_a / "s0001"
        subject_b = self.ds_b / "s0002"
        self.assertEqual(act1._resolve_subject_version(subject_a, self.dataset_path, ann_versions), "v2.1")
        self.assertEqual(act1._resolve_subject_version(subject_b, self.dataset_path, ann_versions), "v3")

    def test_list_unknown_source_returns_none(self):
        """找不到匹配源 → None（退化为只读 segmentations/）。"""
        ann_versions = ["v2.1", "v3"]
        unknown = self.root / "UnknownDataset" / "s0001"
        self.assertEqual(act1._resolve_subject_version(unknown, self.dataset_path, ann_versions), None)

    def test_none_value(self):
        subject = self.ds_a / "s0001"
        self.assertIsNone(act1._resolve_subject_version(subject, self.dataset_path, None))

    def test_same_dir_name_prefix_match(self):
        """审查2a: 两个末级目录名相同的源数据集，靠完整路径前缀匹配，不错配到第一个。"""
        # 两个不同父目录下都有名为 v201 的数据集
        root_a = self.root / "A"
        root_b = self.root / "B"
        ds_a2 = root_a / "v201"
        ds_b2 = root_b / "v201"
        ds_a2.mkdir(parents=True)
        ds_b2.mkdir(parents=True)
        dataset_path = [str(ds_a2), str(ds_b2)]
        ann_versions = ["v2.1", "v3"]
        # ds_b2 下的 case 应取 v3，不应错配到 ds_a2 的 v2.1
        subject_b = ds_b2 / "s0001"
        self.assertEqual(act1._resolve_subject_version(subject_b, dataset_path, ann_versions), "v3")
        subject_a = ds_a2 / "s0001"
        self.assertEqual(act1._resolve_subject_version(subject_a, dataset_path, ann_versions), "v2.1")


class TestVersionParsingEdgeCases(unittest.TestCase):
    """审查1/5: 版本号解析的边界情况。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.subject = Path(self.tmp.name) / "s0001"
        self.subject.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _dir(self, name):
        d = self.subject / name
        d.mkdir(exist_ok=True)
        return d

    def test_multidigit_minor_sorted_as_integer(self):
        """审查1c: 多位数 minor 按整数排序，v2.10 排在 v2.9 之后（降序链中 v2.9 在前）。"""
        self._dir("segmentations_v2.2")
        self._dir("segmentations_v2.9")
        self._dir("segmentations_v2.10")
        chain = act1._version_dirs(self.subject, "v2.10")
        names = [p.name for p in chain]
        # 降序: v2.10, v2.9, v2.2, v2, segmentations（整数序，非字符串序）
        self.assertEqual(names,
                         ["segmentations_v2.10", "segmentations_v2.9",
                          "segmentations_v2.2", "segmentations_v2", "segmentations"])

    def test_invalid_versions_rejected(self):
        """审查1d: 非法版本号一律报错。"""
        for bad in ("v2.1.3", "latest", "v2.a", "v2.1.x", "v-1", "abc", "v", "v.", ".1"):
            with self.assertRaises(ValueError, msg=f"ver={bad!r}"):
                act1._version_dirs(self.subject, bad)

    def test_v2_0_rejected(self):
        """审查1b: vN.0 非法（等价大版本应直接写 v2）。"""
        with self.assertRaises(ValueError):
            act1._version_dirs(self.subject, "v2.0")

    def test_bare_integer_accepted(self):
        """裸整数 '2' 等价 'v2'（大版本）。"""
        self._dir("segmentations_v2")
        chain = act1._version_dirs(self.subject, "2")
        self.assertEqual([p.name for p in chain], ["segmentations_v2", "segmentations"])

    def test_v1x_dir_mapping(self):
        """审查1a/5: v1.2 → segmentations_v1.2/（小版本目录带 v1 前缀），基础版仍是 segmentations/。"""
        self._dir("segmentations_v1.2")
        chain = act1._version_dirs(self.subject, "v1.2")
        self.assertEqual([p.name for p in chain], ["segmentations_v1.2", "segmentations"])

    def test_v1_major_not_segmentations_v1(self):
        """审查1a: 大版本 v1 → segmentations/，绝不构造 segmentations_v1/。"""
        chain = act1._version_dirs(self.subject, "v1")
        self.assertEqual([p.name for p in chain], ["segmentations"])

    def test_scan_ignores_non_version_dirs(self):
        """审查3b: 扫描忽略 segmentations_v2.backup 等非纯数字后缀目录。"""
        self._dir("segmentations_v2.1")
        self._dir("segmentations_v2.backup")
        self._dir("segmentations_v2.1rc")
        minors = act1._scan_minor_versions(self.subject, 2)
        self.assertEqual(minors, {1})

    def test_target_dir_empty_falls_back(self):
        """审查4b/5: 目标版本目录存在但为空 → 器官回退到低版本（告警走回退类，非目录缺失类）。"""
        base = self._dir("segmentations")
        (base / "liver.nii.gz").touch()
        v23 = self._dir("segmentations_v2.3")  # 存在但空
        chain = act1._version_dirs(self.subject, "v2.3")
        record = []
        act1._resolve_organ_path(self.subject, "liver", chain, record=record)
        # v2.3 空, liver 回退到 segmentations/
        self.assertEqual(record[0], ("liver", "segmentations"))


class TestNormalizeAnnotationVersion(unittest.TestCase):
    """审查2c/5: _normalize_annotation_version 长度校验。"""

    def test_scalar_broadcast(self):
        self.assertEqual(framework._normalize_annotation_version("v2", 3), ["v2", "v2", "v2"])

    def test_none_broadcast(self):
        self.assertEqual(framework._normalize_annotation_version(None, 2), [None, None])

    def test_empty_string_normalized_to_none(self):
        self.assertEqual(framework._normalize_annotation_version("", 2), [None, None])

    def test_list_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            framework._normalize_annotation_version(["v2", "v3"], 3)

    def test_list_exact_length(self):
        self.assertEqual(
            framework._normalize_annotation_version(["v2.1", "v3"], 2),
            ["v2.1", "v3"])


if __name__ == "__main__":
    unittest.main()
