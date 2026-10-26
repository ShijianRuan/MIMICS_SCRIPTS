"""Workflow regressions with documented Mimics boundaries and real file/job logic.

No Mimics license, GPU, model downloads or patient data are used.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools'), str(ROOT/'runtime_py35')]
spec = importlib.util.spec_from_file_location('review_support', ROOT/'tests/test_all.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
import import_undo_mimics as undo
import mimics_export as export
import nnunet_jobs as jobs

class MaskContainer(list):
    def delete(self, mask):
        self.remove(mask)

class TestDocumentedUndo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)/'case.mcs'
        self.project.write_bytes(b'human project')
        self.receipt = Path(self.tmp.name)/'receipt.json'
        self.receipt.write_text('{}')
        # S04: the delete path only runs for a mask the receipt can PROVE
        # unedited, so the live mask exposes the same voxel content recorded
        # in its sha256 snapshot.
        import hashlib
        self.content = b'unedited-voxels'
        self.mask = types.SimpleNamespace(
            name='liver', guid='owned', get_voxel_buffer=lambda: self.content)
        self.masks = MaskContainer([self.mask])
        self.payload = {'schema_version':undo.RECEIPT_SCHEMA_VERSION,
                        'mcs_path':str(self.project),'created_masks':['liver'],
                        'created_masks_v2':[{'name':'liver','guid':'owned',
                                             'sha256':'sha256:'+hashlib.sha256(self.content).hexdigest()}]}
    def environment(self):
        stack = __import__('contextlib').ExitStack()
        stack.enter_context(mock.patch.object(undo,'find_latest_receipt',return_value=(str(self.receipt),self.payload)))
        stack.enter_context(mock.patch.object(undo,'_open_project',return_value=True))
        # Images intentionally have only the documented linked_objects.
        stack.enter_context(mock.patch.object(undo.mimics.data,'images',[types.SimpleNamespace(linked_objects=self.masks)]))
        stack.enter_context(mock.patch.object(undo.mimics.data,'masks',self.masks))
        stack.enter_context(mock.patch.object(undo.runtime_common,'execute_mimics_transaction',side_effect=lambda api,op,title:op()))
        return stack
    def test_owned_mask_resolution_needs_no_image_masks_attribute(self):
        with self.environment():
            found, ambiguous = undo._resolve_owned_masks(self.payload)
        self.assertEqual({0:self.mask},found)
        self.assertEqual([],ambiguous)
    def test_delete_uses_container_api_without_mask_delete_method(self):
        with self.environment():
            deleted, ambiguous, edited = undo._delete_owned_masks(self.payload)
        self.assertEqual(['liver'],deleted)
        self.assertEqual([],edited)
        self.assertEqual([],self.masks)
    def test_confirm_uses_answered_question_not_message_return(self):
        with self.environment(), mock.patch.object(undo.mimics.dialogs,'message_box',return_value=None), mock.patch.object(undo.mimics.dialogs,'question_box',return_value='撤销') as question:
            code = undo.undo_last_import(confirm=True)
        self.assertEqual(0,code)
        self.assertEqual(1,question.call_count)
        self.assertFalse(self.receipt.exists())
    def test_cancel_preserves_receipt_masks_and_project(self):
        with self.environment(), mock.patch.object(undo.mimics.dialogs,'message_box',return_value=None), mock.patch.object(undo.mimics.dialogs,'question_box',return_value='取消'):
            code = undo.undo_last_import(confirm=True)
        self.assertEqual(2,code)
        self.assertEqual([self.mask],self.masks)
        self.assertTrue(self.receipt.exists())
        self.assertEqual(b'human project',self.project.read_bytes())

class TestSelectedExportGrid(unittest.TestCase):
    def test_filtered_export_uses_selected_image_not_project_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_a = types.SimpleNamespace(guid='A',logical_dimensions=[2,2,2])
            image_b = types.SimpleNamespace(guid='B',logical_dimensions=[3,4,5])
            masks = [types.SimpleNamespace(name='other',image=image_a),types.SimpleNamespace(name='target',image=image_b)]
            def stream(mask,path):
                Path(path).write_bytes(bytes(60))
            with mock.patch.object(export.mimics.data,'masks',masks), mock.patch.object(export,'_derive_mimics_voxel_to_ras_matrix',side_effect=lambda image,shape:[[1,0,0,100 if image is image_b else 0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), mock.patch.object(export,'_stream_mask_to_file',side_effect=stream):
                manifest=export.export_masks_to_buffers(tmp,mask_names=['target'])
            self.assertEqual([3,4,5],manifest['mimics_shape'])
            self.assertEqual(100,manifest['mimics_voxel_to_ras_matrix'][0][3])
    def test_empty_selection_does_not_claim_unrelated_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            masks=[types.SimpleNamespace(name='other',image=types.SimpleNamespace(guid='A',logical_dimensions=[2,2,2]))]
            with mock.patch.object(export.mimics.data,'masks',masks):
                manifest=export.export_masks_to_buffers(tmp,mask_names=['missing'])
            self.assertIsNone(manifest['mimics_shape'])
            self.assertEqual([],manifest['masks'])

    def test_filtered_conversion_keeps_selected_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_a = types.SimpleNamespace(guid='A',logical_dimensions=[2,2,2])
            image_b = types.SimpleNamespace(guid='B',logical_dimensions=[3,4,5])
            masks = [types.SimpleNamespace(name='other',image=image_a),types.SimpleNamespace(name='target',image=image_b)]
            def metadata(image,key,default=''):
                if key == export.SOURCE_IMAGE_PATH_METADATA:
                    return 'source-B.nii.gz' if image is image_b else 'source-A.nii.gz'
                return default
            def stream(mask,path):
                Path(path).write_bytes(bytes(60))
            with mock.patch.object(export.mimics.data,'masks',masks), mock.patch.object(export.mimics.data.images,'get_active',return_value=image_a), mock.patch.object(export,'_metadata_get',side_effect=metadata), mock.patch.object(export,'_derive_mimics_voxel_to_ras_matrix',return_value=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), mock.patch.object(export,'_stream_mask_to_file',side_effect=stream):
                params,manifest = export._export_masks_and_build_params(tmp,[0,1,2],[False]*3,str(Path(tmp)/'work'),mask_names=['target'])
                with self.assertRaisesRegex(RuntimeError,'different source image'):
                    export._export_masks_and_build_params(tmp,[0,1,2],[False]*3,str(Path(tmp)/'work'),mask_names=['target'],source_image_path='source-A.nii.gz')
            self.assertEqual('source-B.nii.gz',params['source_image_path'])
            self.assertEqual([3,4,5],manifest['mimics_shape'])

class TestJobPreflight(unittest.TestCase):
    def test_missing_remote_profile_creates_no_launching_ghost(self):
        with tempfile.TemporaryDirectory() as tmp:
            values={'workspace':tmp,'operation':'infer','execution_backend':'remote','remote_profile_id':'','job_id':'bad_remote'}
            with mock.patch.object(jobs,'_launch') as launch:
                with self.assertRaises(RuntimeError):jobs.create_job(values)
            self.assertFalse((Path(tmp)/'jobs'/'bad_remote').exists())
            launch.assert_not_called()

    def test_project_voxels_remote_training_rejected_before_job_dir(self):
        # D28：远程上传路径尚未覆盖工程导出的图像 raw，必须在任何 job 工件
        # 产生之前拒绝，不留 launching 幽灵。nnU-Net 与 FlexiCT 两个创建器
        # 都要拦（FlexiCT 训练数据准备复用同一条 project_voxels 管线）。
        import flexict_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            values={'workspace':tmp,'operation':'train','label_source':'project_voxels','mcs_dir':tmp,'labels':[{'name':'target','id':1}],'execution_backend':'remote','remote_profile_id':'srv','job_id':'pv_remote'}
            with mock.patch.object(jobs,'_launch') as launch:
                with self.assertRaisesRegex(RuntimeError,'工程体素训练模式首期仅支持本机'):jobs.create_job(values)
            self.assertFalse((Path(tmp)/'jobs'/'pv_remote').exists())
            launch.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            values={'workspace':tmp,'operation':'train','task_name':'Kidney','label_name':'kidney_left','cases':['case01','case02'],'label_source':'project_voxels','mcs_dir':tmp,'dataset_id':750,'configuration':'2d','epochs':2,'val_cases':1,'mimics_exe':'','execution_backend':'remote','remote_profile_id':'srv'}
            with mock.patch('flexict_pipeline.subprocess.Popen') as popen:
                with self.assertRaisesRegex(RuntimeError,'工程体素训练模式首期仅支持本机'):flexict_pipeline.create_flexict_job(values)
            self.assertEqual([], list(Path(tmp).rglob('*train*')))
            popen.assert_not_called()

    def test_project_voxels_active_learning_rejected_before_job_dir(self):
        # D28：主动学习排序硬性依赖 dataset_root 下的原图病例，工程体素模式
        # 没有该数据集——同样必须在 mkdir 之前拒绝。FlexiCT 的主动学习
        # 也走 create_flexict_job，同样必须拦下。
        import flexict_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            values={'workspace':tmp,'operation':'active_learning','label_source':'project_voxels','mcs_dir':tmp,'execution_backend':'local','job_id':'pv_al'}
            with mock.patch.object(jobs,'_launch') as launch:
                with self.assertRaisesRegex(RuntimeError,'暂不支持主动学习'):jobs.create_job(values)
            self.assertFalse((Path(tmp)/'jobs'/'pv_al').exists())
            launch.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            values={'workspace':tmp,'operation':'active_learning','label_source':'project_voxels','mcs_dir':tmp,'execution_backend':'local'}
            with mock.patch('flexict_pipeline.subprocess.Popen') as popen:
                with self.assertRaisesRegex(RuntimeError,'暂不支持主动学习'):flexict_pipeline.create_flexict_job(values)
            self.assertEqual([], list(Path(tmp).rglob('*active_learning*')))
            popen.assert_not_called()

    def test_flexict_remote_preflights_reject_before_job_dir(self):
        # R97 P3-1/P3-3：FlexiCT 创建器此前在 status.json 写盘之后才查
        # remote_profile_id，且不拒绝 active_learning+remote。两种都会留下
        # 没有 PID 可对账的 "launching" 幽灵任务。必须与 nnunet_jobs.create_job
        # 一样在任何 job 工件产生之前拒绝。
        import flexict_pipeline
        train_values = {'workspace': '', 'operation': 'train', 'task_name': 'Kidney',
                        'label_name': 'kidney_left', 'cases': ['case01', 'case02'],
                        'dataset_id': 750, 'configuration': '2d', 'epochs': 2,
                        'val_cases': 1, 'mimics_exe': ''}
        with tempfile.TemporaryDirectory() as tmp:
            train_values['workspace'] = tmp
            with mock.patch('flexict_pipeline.subprocess.Popen') as popen:
                with self.assertRaisesRegex(
                        RuntimeError, 'Remote training server was not selected'):
                    flexict_pipeline.create_flexict_job(
                        dict(train_values, execution_backend='remote',
                             remote_profile_id=''))
            self.assertEqual([], list(Path(tmp).rglob('status.json')))
            popen.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('flexict_pipeline.subprocess.Popen') as popen:
                with self.assertRaisesRegex(
                        RuntimeError, '主动学习任务首期仅支持本机'):
                    flexict_pipeline.create_flexict_job(
                        {'workspace': tmp, 'operation': 'active_learning',
                         'execution_backend': 'remote', 'remote_profile_id': 'srv'})
            self.assertEqual([], list(Path(tmp).rglob('status.json')))
            popen.assert_not_called()

    def test_flexict_launch_failure_marks_failed_not_launching(self):
        # R97 P3-2：Popen 失败必须把 status 落成 failed/launch_failed。
        # 此前异常直接抛出，status.json 停在 "launching"，reconcile 找不到
        # 任何 PID，任务永远无人认领。
        import flexict_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            values = {'workspace': tmp, 'operation': 'train',
                      'task_name': 'Kidney', 'label_name': 'kidney_left',
                      'cases': ['case01'], 'dataset_id': 750,
                      'configuration': '2d', 'epochs': 2, 'val_cases': 1,
                      'mimics_exe': '', 'label_source': 'project_voxels',
                      'mcs_dir': tmp, 'execution_backend': 'local'}
            with mock.patch('flexict_pipeline.subprocess.Popen',
                            side_effect=OSError('boom')):
                with self.assertRaises(OSError):
                    flexict_pipeline.create_flexict_job(values)
            status_files = list(Path(tmp).rglob('status.json'))
            self.assertEqual(1, len(status_files))
            payload = json.loads(status_files[0].read_text(encoding='utf-8'))
            self.assertEqual('failed', payload['status'])
            self.assertEqual('launch_failed', payload['phase'])
            self.assertIn('boom', str(payload.get('error')))

class TestBatchJobIdentity(unittest.TestCase):
    def test_cli_chinese_cases_get_distinct_ids_in_same_second(self):
        import mimics_batch_cli as cli
        import nnunet_common
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cases';root.mkdir()
            for case in ('病例甲','病例乙'):
                path=root/case;path.mkdir();(path/'ct.nii.gz').write_bytes(b'x')
            model={'model_id':'m','task_id':'t','manifest_path':str(Path(tmp)/'manifest.json')}
            created=[]
            def launch(request):
                normalized=nnunet_common.normalize_request(request)
                self.assertNotIn(normalized['job_id'],created)
                created.append(normalized['job_id'])
                return {'status_path':'status.json'}
            args=types.SimpleNamespace(workspace=str(Path(tmp)/'ws'),model_id='m',image_root=str(root),cases=None,output_dir=str(Path(tmp)/'out'),disable_tta=True,use_cpu=True,source_modality='CT')
            with mock.patch.object(nnunet_common,'load_models',return_value=[model]), mock.patch.object(nnunet_common,'model_usability',return_value=(True,'')), mock.patch.object(jobs,'create_job',side_effect=launch), mock.patch.object(cli.time,'strftime',return_value='20261011T120000'):
                self.assertEqual(0,cli.cmd_predict(args))
            self.assertEqual(2,len(created))

class TestOverlayRollback(unittest.TestCase):
    def run_failure(self, delete_fails=False):
        import flexict_mimics as flex
        with tempfile.TemporaryDirectory() as tmp:
            bridge=Path(tmp)/'bridge';bridge.mkdir()
            result=bridge/'result.json';result.write_text('{"status":"ok"}')
            alive=MaskContainer();written=[];marked=[]
            def new(title):
                mask=types.SimpleNamespace(name=title);alive.append(mask);return mask
            def write(*args):
                if written:raise RuntimeError('second band write failed')
                written.append(args)
            if delete_fails:
                def fail(mask):raise RuntimeError('host delete failed')
                alive.delete=fail
            trans={'request':{'_request_path':str(Path(tmp)/'req.json'),'_job_dir':tmp,'case_id':'c'},'result_path':str(result),'bridge_root':str(bridge),'masks':[]}
            rows=[('high',{'output_path':'x','mimics_shape':[2,2,2]}),('moderate',{'output_path':'y','mimics_shape':[2,2,2]})]
            with mock.patch.object(flex,'_al_verify_conversion_buffers',return_value=(rows,'')),mock.patch.object(flex,'_al_target_still_matches',return_value=''),mock.patch.object(flex.mimics_mask_apply,'_new_prediction_mask',side_effect=new),mock.patch.object(flex.mimics_mask_apply,'_set_mask_from_u8',side_effect=write),mock.patch.object(flex.mimics.data,'masks',alive),mock.patch.object(flex,'_al_mark_request',side_effect=lambda p,state,detail:marked.append((state,detail))):
                flex._al_finish_conversion(trans)
            return alive,marked
    def test_partial_overlay_is_deleted_using_container_api(self):
        alive,marked=self.run_failure()
        self.assertEqual([],alive)
        self.assertEqual('failed',marked[0][0])
        self.assertIn('rolled back',marked[0][1])
    def test_failed_rollback_reports_remaining_masks_honestly(self):
        alive,marked=self.run_failure(True)
        self.assertEqual(2,len(alive))
        self.assertIn('Rollback incomplete',marked[0][1])
        self.assertIn('high',marked[0][1])
        self.assertIn('moderate',marked[0][1])

class TestPortableGridExport(unittest.TestCase):
    """R94-2：分发场景——A 转好 .mcs 发给 B，B 机器上没有原图。
    管线创建的工程把 A 的绝对路径记进元数据并携带导入时的网格快照
    （shape + voxel-to-RAS 矩阵）。导出必须靠快照落回原始网格，
    而不是因路径已死而拒绝运行。"""

    MATRIX = '[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # The recorded source path only exists on the importer's machine.
        self.dead_source = self.root / 'importer_machine' / 'ct.nii.gz'
        self.image = types.SimpleNamespace(guid='A', logical_dimensions=[2, 2, 2])
        self.mask = types.SimpleNamespace(name='liver', image=self.image)

    def metadata(self, snapshot=True):
        values = {export.SOURCE_IMAGE_PATH_METADATA: str(self.dead_source)}
        if snapshot:
            values[export.SOURCE_IMAGE_SHAPE_METADATA] = '[2, 2, 2]'
            values[export.SOURCE_VOXEL_TO_RAS_MATRIX_METADATA] = self.MATRIX
        def read(image, key, default=''):
            return values.get(key, default)
        return read

    def patches(self, metadata, confirm=None, monitor=None, bridge=None):
        stack = __import__('contextlib').ExitStack()
        add = stack.enter_context
        add(mock.patch.object(export.mimics.data, 'masks', [self.mask]))
        add(mock.patch.object(export.mimics.data.images, 'get_active', return_value=self.image))
        add(mock.patch.object(export, '_metadata_get', side_effect=metadata))
        add(mock.patch.object(export, '_derive_mimics_voxel_to_ras_matrix',
                              return_value=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        add(mock.patch.object(export.runtime_common, 'try_acquire_local_operation', return_value='token'))
        add(mock.patch.object(export.runtime_common, 'release_local_operation', lambda *a: None))
        add(mock.patch.object(export, '_current_project_case_id', return_value='s0001'))
        add(mock.patch.object(export, '_current_project_path', return_value=str(self.root / 's0001.mcs')))
        add(mock.patch.object(export, '_project_root', return_value=self.tmp.name))
        if confirm is not None:
            add(mock.patch.object(export, '_confirm_degraded_export', side_effect=confirm))
        if monitor is not None:
            add(mock.patch.object(export, '_start_foreground_export_monitor', side_effect=monitor))
        add(mock.patch.object(export, '_stream_mask_to_file',
                              side_effect=lambda mask, path: (Path(path).write_bytes(b'\x01' * 8), 8)[1]))
        if bridge is not None:
            add(mock.patch.object(export, '_launch_bridge_background', side_effect=bridge))
        return stack

    def run_export(self, snapshot, confirm_result=None):
        """Drive _start_current_project_export through to the bridge params."""
        confirm_calls = []
        def confirm(count=1):
            confirm_calls.append(count)
            return confirm_result
        captured = {}
        def monitor_start(m):
            captured['monitor'] = m
            return True
        class Process(object):
            pid = 4321
        def bridge_launch(params, job_dir):
            captured['params'] = params
            return Process()
        with self.patches(self.metadata(snapshot), confirm=confirm,
                          monitor=monitor_start, bridge=bridge_launch):
            descriptor = export._start_current_project_export(
                str(self.root / 'case'), str(self.dead_source),
                str(self.root / 'out'), [0, 1, 2], [False] * 3)
            # One tick reads the mask buffer, the next builds the bridge params.
            export._foreground_export_tick(captured['monitor'])
            export._foreground_export_tick(captured['monitor'])
        return captured, confirm_calls, descriptor

    def test_dead_source_with_snapshot_exports_on_original_grid(self):
        captured, confirm_calls, descriptor = self.run_export(snapshot=True)
        self.assertEqual([], confirm_calls, 'snapshot present: no degraded confirm')
        self.assertEqual('export', descriptor['kind'])
        params = captured['params']
        self.assertEqual(str(self.dead_source), params['source_image_path'])
        self.assertEqual('[2, 2, 2]', params['source_image_shape'])
        self.assertEqual(self.MATRIX, params['source_voxel_to_ras_matrix'])
        self.assertTrue(params['require_source_geometry'])

    def test_dead_source_without_snapshot_requires_user_confirmation(self):
        with self.assertRaisesRegex(RuntimeError, 'original source image'):
            self.run_export(snapshot=False, confirm_result=False)
        # Cancelled: no export monitor may have been started at all.
        # (run_export never reaches the tick calls when start raises.)

    def test_dead_source_without_snapshot_degrades_only_on_user_accept(self):
        captured, confirm_calls, descriptor = self.run_export(
            snapshot=False, confirm_result=True)
        self.assertEqual([1], confirm_calls)
        params = captured['params']
        self.assertEqual('', params['source_image_path'])
        self.assertNotIn('source_image_shape', params)
        self.assertFalse(params['require_source_geometry'])

    def test_setup_ui_portable_flag_only_with_snapshot(self):
        import io_setup_mimics as setup
        contexts = []
        def fake_launch(mode, python_exe, context, callback, timeout_seconds=3600):
            contexts.append(context)
            return 0
        for snapshot, expected in ((True, True), (False, False)):
            contexts[:] = []
            with self.patches(self.metadata(snapshot)), \
                    mock.patch.object(export, '_python_exe', return_value='py'), \
                    mock.patch.object(setup, 'launch', side_effect=fake_launch):
                export._launch_external_export_setup()
            self.assertEqual(expected, contexts[0]['source_portable_grid'])
            self.assertEqual(str(self.dead_source), contexts[0]['source_initial'])

class TestProjectBufferPrediction(unittest.TestCase):
    """D20：无原图机子上的预测。分发出去的 .mcs 没有源文件，但工程自带的
    体素 buffer + 导入时网格快照可以取代它。失败路径优先：远程必须在任何
    job 工件产生之前拒绝；sha256 失守不得留下半 custody 任务目录；buffer
    模式回写不得再依赖数据集根（挂死修复）；无快照工程报错逐字不变。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.grid = {
            'target_shape': [2, 2, 2],
            'target_voxel_to_ras_matrix': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        }
        self.snapshot = {
            'source_image_path': str(self.root / 'importer_machine' / 'ct.nii.gz'),
            'source_shape': [2, 2, 2],
            'source_modality': 'CT',
            'source_voxel_to_ras_matrix': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        }

    def write_raw(self, sha=None):
        import hashlib
        data = b'\x01' * 16
        raw = self.root / 'setup' / 'case_raw.raw'
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(data)
        payload = {
            'path': str(raw),
            'shape': [2, 2, 2],
            'dtype': 'int16',
            'sha256': sha or hashlib.sha256(data).hexdigest(),
            'mimics_voxel_to_ras_matrix': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        }
        return payload, raw

    def test_buffer_remote_rejected_before_any_job_artifact(self):
        import flexict_pipeline
        for creator in (jobs.create_job, flexict_pipeline.create_flexict_job):
            values = {
                'workspace': self.tmp.name, 'operation': 'infer',
                'execution_backend': 'remote', 'remote_profile_id': 'srv',
                'job_id': 'bad_buffer_remote',
                'project_image_export': {'path': str(self.root / 'never.raw')},
            }
            with mock.patch('subprocess.Popen') as popen:
                with self.assertRaises(RuntimeError) as ctx:
                    creator(values)
            self.assertIn('仅支持本机执行', str(ctx.exception))
            self.assertEqual([], list(self.root.rglob('bad_buffer_remote*')))
            popen.assert_not_called()

    def test_buffer_sha256_mismatch_leaves_no_half_custody_job(self):
        payload, raw = self.write_raw(sha='deadbeef')
        values = {
            'workspace': self.tmp.name, 'operation': 'infer',
            'job_id': 'custody_bad', 'project_image_export': payload,
        }
        with mock.patch.object(jobs, '_launch',
                               return_value=types.SimpleNamespace(pid=1234)):
            with self.assertRaises(RuntimeError) as ctx:
                jobs.create_job(values)
        self.assertIn('sha256', str(ctx.exception))
        self.assertEqual([], list(self.root.rglob('custody_bad*')))
        self.assertTrue(raw.exists(), 'the setup raw must stay for a retry')

    def test_buffer_custody_moves_raw_into_job_input(self):
        import json
        payload, raw = self.write_raw()
        values = {
            'workspace': self.tmp.name, 'operation': 'infer',
            'job_id': 'custody_ok', 'project_image_export': payload,
        }
        with mock.patch.object(jobs, '_launch',
                               return_value=types.SimpleNamespace(pid=1234)):
            result = jobs.create_job(values)
        self.assertFalse(raw.exists())
        request = json.loads(
            (Path(result['job_dir']) / 'request.json').read_text())
        moved = Path(request['project_image_export']['path'])
        self.assertEqual(
            moved, Path(result['job_dir']) / 'input' / 'project_image.raw')
        self.assertTrue(moved.exists())

    def test_monitor_buffer_mode_skips_dataset_root_check(self):
        import mimics_mask_apply
        monitor = {
            'project_image_export': {'path': 'x.raw'}, 'ts_root': '',
            'case_id': 's0001',
            'launch_project_path': str(self.root / 's0001.mcs'),
            'target_grid': dict(self.grid),
        }
        def boom(value):
            raise AssertionError(
                'buffer mode must not infer the case from an empty ts_root')
        with mock.patch.object(mimics_mask_apply, '_infer_case_id',
                               side_effect=boom), \
                mock.patch.object(mimics_mask_apply, '_current_project_path',
                                  return_value=str(self.root / 's0001.mcs')), \
                mock.patch.object(mimics_mask_apply, '_active_live_grid_payload',
                                  return_value=dict(self.grid)):
            self.assertEqual((True, ''),
                             mimics_mask_apply._monitor_target_is_open(monitor))
        with mock.patch.object(mimics_mask_apply, '_infer_case_id',
                               side_effect=boom), \
                mock.patch.object(mimics_mask_apply, '_current_project_path',
                                  return_value=str(self.root / 'other.mcs')), \
                mock.patch.object(mimics_mask_apply, '_active_live_grid_payload',
                                  return_value=dict(self.grid)):
            ok, message = mimics_mask_apply._monitor_target_is_open(monitor)
        self.assertFalse(ok)
        self.assertIn('original project', message)

    def test_monitor_non_buffer_mode_still_requires_the_source_case(self):
        import mimics_mask_apply
        monitor = {
            'ts_root': str(self.root / 'ts'), 'case_id': 's0001',
            'launch_project_path': '', 'target_grid': {},
        }
        with mock.patch.object(mimics_mask_apply, '_infer_case_id',
                               return_value='s0002'):
            ok, message = mimics_mask_apply._monitor_target_is_open(monitor)
        self.assertFalse(ok)
        self.assertIn('Open the source case', message)

    def context_patches(self, module, export_side_effect, snapshot=True):
        stack = __import__('contextlib').ExitStack()
        add = stack.enter_context
        add(mock.patch.object(module, '_selected_masks', return_value=[]))
        add(mock.patch.object(module, '_setup_root',
                              return_value=str(self.root / 'setup')))
        add(mock.patch.object(module, '_active_image_masks', return_value=[]))
        add(mock.patch.object(module.mimics_mask_apply,
                              '_resolve_prediction_context',
                              return_value=('', 's0001', '')))
        add(mock.patch.object(
            module.mimics_mask_apply, '_active_source_geometry_payload',
            return_value=dict(self.snapshot) if snapshot else None))
        add(mock.patch.object(module.mimics_mask_apply,
                              '_export_project_image_for_prediction',
                              side_effect=export_side_effect))
        add(mock.patch.object(module.mimics_mask_apply, '_active_live_grid_payload',
                              return_value=dict(self.grid)))
        add(mock.patch.object(module.mimics_mask_apply, '_current_project_path',
                              return_value=str(self.root / 's0001.mcs')))
        return stack

    def test_dead_source_with_snapshot_falls_back_to_project_buffer(self):
        import flexict_mimics
        import nnunet_mimics
        for module in (nnunet_mimics, flexict_mimics):
            calls = []
            def fake_export(setup_root, snapshot):
                calls.append((setup_root, snapshot))
                return {'path': str(self.root / 'setup' / 'case.raw')}
            with self.context_patches(module, fake_export):
                context = module._prediction_context()
            self.assertEqual([str(self.root / 'setup')],
                             [call[0] for call in calls])
            self.assertEqual(dict(self.snapshot), calls[0][1])
            self.assertEqual({'path': str(self.root / 'setup' / 'case.raw')},
                             context['project_image_export'])
            self.assertEqual(self.snapshot['source_shape'],
                             context['source_geometry_expected']['source_shape'])
            self.assertEqual('', context['source_image_path'])

    def test_dead_source_without_snapshot_keeps_original_error(self):
        import flexict_mimics
        import nnunet_mimics
        for module in (nnunet_mimics, flexict_mimics):
            calls = []
            def fake_export(setup_root, snapshot):
                calls.append((setup_root, snapshot))
                return None
            with self.context_patches(module, fake_export, snapshot=False):
                with self.assertRaises(RuntimeError) as ctx:
                    module._prediction_context()
            self.assertEqual([], calls, 'no raw may be exported without a snapshot')
            self.assertIn('原始源影像', str(ctx.exception))
            self.assertIn('01_Import_Data', str(ctx.exception))

if __name__=='__main__':unittest.main()
