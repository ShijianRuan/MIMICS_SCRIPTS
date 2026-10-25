"""Workflow regressions with documented Mimics boundaries and real file/job logic.

No Mimics license, GPU, model downloads or patient data are used.
"""
import importlib.util
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

if __name__=='__main__':unittest.main()
