"""Safe worker reuse and physical-input cache contracts; no models or GPU."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'runtime_py35'), str(ROOT/'tools'), str(ROOT)]
import fake_mimics_flow_test as support
fake = support.install_fake_mimics()
import nninteractive_mimics as runtime
import nninteractive_worker_identity as ownership


class WorkerSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root/'model'
        (self.model/'fold_0').mkdir(parents=True)
        (self.model/'fold_0/checkpoint_final.pth').write_bytes(b'fixture weights')
        self.config = dict(runtime_work_dir=str(self.root/'runtime'), model_dir=str(self.model))
        self.image = fake.reset_scene(image_shape=(2, 2, 2))
        fake.file.project_path = str(self.root/'case.mcs')
        self.live = {1234567}
        self.markers = {1234567:'created-1', os.getpid():'parent-created'}
        self.patches = [mock.patch.object(runtime.runtime_common,'process_exists',side_effect=lambda pid: int(pid or 0) in self.live),
                        mock.patch.object(runtime.runtime_common,'process_start_marker',side_effect=lambda pid:self.markers.get(int(pid),''))]
        for patch in self.patches:
            patch.start();self.addCleanup(patch.stop)
        runtime._ASYNC_IMAGE_WORKERS.clear();runtime._ASYNC_MONITORS.clear()
        self.addCleanup(runtime._ASYNC_IMAGE_WORKERS.clear)
        self.addCleanup(runtime._ASYNC_MONITORS.clear)
        self.addCleanup(ownership._session()['handles'].clear)

    def worker(self, image=None, name='image_worker_fixture', status='ready'):
        image = image or self.image
        folder = self.root/'runtime/async_jobs'/name
        folder.mkdir(parents=True)
        identity = ownership.new_identity()
        (folder/'initialize.json').write_text(json.dumps(dict(worker_identity=identity)))
        process = types.SimpleNamespace(pid=1234567,poll=lambda:None)
        row = dict(worker_dir=str(folder),pid=process.pid,cache_key=runtime._worker_cache_key(self.config,image),
                   image_guid=image.guid,image_name=image.name,model_identity='official',
                   created_at_epoch=1,shape=[2,2,2],model_dir=str(self.model),image_source='mimics_buffer',
                   input_identity=runtime._image_worker_input_identity(self.config,image,str(self.model)))
        row.update(ownership.attach_process(str(folder),process,identity))
        (folder/'worker_status.json').write_text(json.dumps(dict(status=status,pid=process.pid,worker_token=row['worker_token'])))
        (folder/'image_worker.json').write_text(json.dumps(row))
        return row

    def test_disk_reuse_survives_module_reload_for_unchanged_owned_image(self):
        row=self.worker()
        importlib.reload(runtime)
        self.assertEqual(1,runtime._adopt_disk_image_workers(self.config))
        with mock.patch.object(runtime,'_runtime_paths',side_effect=AssertionError('must reuse')):
            current=runtime._get_or_start_image_worker(self.config,self.image,str(self.root/'runtime/async_jobs'))
        self.assertEqual(row['worker_token'],current['worker_token'])

    def test_changed_grid_content_and_model_never_reuse_old_input(self):
        for change in ('grid','content','model'):
            with self.subTest(change=change):
                self.image=fake.reset_scene(image_shape=(2,2,2))
                row=self.worker(name='image_worker_'+change)
                runtime._ASYNC_IMAGE_WORKERS[row['cache_key']]=row
                if change=='grid':
                    self.image.get_voxel_center=lambda index:[100-float(index[0]),-float(index[1]),float(index[2])]
                elif change=='content':
                    self.image._buffer=support.FakeVoxelBuffer((2,2,2),b'\x07'*8)
                else:
                    (self.model/'fold_0/checkpoint_final.pth').write_bytes(b'changed checkpoint')
                with mock.patch.object(runtime,'_runtime_paths',side_effect=RuntimeError('cold path')), \
                     mock.patch.object(runtime.runtime_common,'terminate_process_async') as terminate:
                    with self.assertRaisesRegex(RuntimeError,'cold path'):
                        runtime._get_or_start_image_worker(self.config,self.image,str(self.root/'runtime/async_jobs'))
                self.assertEqual(1,terminate.call_count)
                runtime._ASYNC_IMAGE_WORKERS.clear()

    def test_changed_input_during_unconsumed_result_is_not_retired(self):
        row=self.worker(status='running');runtime._ASYNC_IMAGE_WORKERS[row['cache_key']]=row
        self.image._buffer=support.FakeVoxelBuffer((2,2,2),b'\x07'*8)
        with mock.patch.object(runtime.runtime_common,'terminate_process_async') as terminate:
            with self.assertRaisesRegex(RuntimeError,'旧预测尚待处理'):
                runtime._get_or_start_image_worker(self.config,self.image,str(self.root/'runtime/async_jobs'))
        terminate.assert_not_called()

    def test_entry_discovers_busy_other_image_before_creating_masks(self):
        self.worker(status='running')
        fake.reset_scene(image_shape=(2,2,2))
        with mock.patch.object(runtime.runtime_common,'active_local_operation',return_value=None), \
             mock.patch.object(runtime.runtime_common,'active_resource_lock',return_value=None), \
             mock.patch.object(runtime,'_select_session_masks') as select, \
             mock.patch.object(runtime,'_prewarm_async_image_worker') as prewarm:
            self.assertEqual(1,runtime._run_with_config(self.config))
        select.assert_not_called();prewarm.assert_not_called()

    def test_foreign_owner_recycled_pid_and_missing_marker_are_not_authority(self):
        row=self.worker()
        for kind in ('foreign','recycled','missing'):
            with self.subTest(kind=kind):
                altered=dict(row)
                if kind=='foreign':altered['owner_token']='another-Mimics'
                elif kind=='recycled':self.markers[1234567]='replacement-process'
                else:
                    self.markers[1234567]='';ownership._session()['handles'].clear()
                runtime._ASYNC_IMAGE_WORKERS['old']=altered
                with mock.patch.object(runtime.runtime_common,'terminate_process_async') as terminate:
                    runtime._retire_idle_image_workers(self.config,keep_image_guid='different')
                terminate.assert_not_called()
                self.assertFalse((Path(row['worker_dir'])/'close.json').exists())

    def test_bad_schema_is_skipped_without_suppressing_good_worker(self):
        good=self.worker()
        folder=self.root/'runtime/async_jobs/image_worker_bad';folder.mkdir()
        broken=dict(good,created_at_epoch='bad clock')
        (folder/'image_worker.json').write_text(json.dumps(broken))
        self.assertEqual(1,runtime._adopt_disk_image_workers(self.config))
        self.assertEqual([good['cache_key']],list(runtime._ASYNC_IMAGE_WORKERS))

    def test_changed_image_result_never_reaches_mask_write(self):
        mask=support.FakeMask('manual',image=self.image,array=support._u8_buffer((2,2,2),1))
        fake.data.masks.append(mask)
        folder=self.root/'job';folder.mkdir()
        state=dict(_job_dir=str(folder),pending_sequence=1,image_guid=self.image.guid,
                   target_guid=mask.guid,shape=[2,2,2],launch_image_identity=runtime._live_image_identity(self.image))
        # The real launch proof is written to JSON and loaded on the next
        # menu invocation; unchanged input must still pass that round trip.
        state=json.loads(json.dumps(state))
        result=Path(runtime._async_result_path(state,1));result.parent.mkdir(exist_ok=True)
        result.write_text(json.dumps(dict(status='refined')))
        with mock.patch.object(runtime,'_mask_sha256',side_effect=RuntimeError('passed input guard')):
            with self.assertRaisesRegex(RuntimeError,'passed input guard'):
                runtime._handle_async_result(self.image,mask,state)
        self.image._buffer=support.FakeVoxelBuffer((2,2,2),b'\x07'*8)
        with mock.patch.object(runtime,'_set_mask_from_u8') as apply, \
             mock.patch.object(runtime,'_close_async_job') as close:
            self.assertEqual('ready',runtime._handle_async_result(self.image,mask,state))
        apply.assert_not_called()
        self.assertEqual('image_input_changed',close.call_args.args[2])
        self.assertEqual(8,mask.number_of_pixels)

    def test_unobservable_source_revision_cannot_authorize_reuse(self):
        proof=runtime._image_worker_input_identity(self.config,self.image,str(self.model))
        proof['source']=['//server/image.nii.gz','network_unverified']
        self.assertFalse(runtime._worker_input_revision_verified(proof,'source_nifti'))
        self.assertTrue(runtime._worker_input_revision_verified(proof,'mimics_buffer'))

    def test_singleton_axis_keeps_measured_spacing_or_refuses_unknown_grid(self):
        image=fake.reset_scene(image_shape=(2,2,1))
        image.get_voxel_center=lambda index:[float(index[0]),float(index[1]),5*float(index[2])]
        self.assertEqual(5,runtime._live_image_identity(image)['grid'][2][2])
        def bounded(index):
            if index[2]:raise IndexError('Mimics API requires an in-bounds index')
            return index
        image.get_voxel_center=bounded
        with self.assertRaisesRegex(RuntimeError,'物理网格'):
            runtime._live_image_identity(image)


class BridgeOwnership(unittest.TestCase):
    def test_image_lease_excludes_second_context_and_constructor_failure_releases_it(self):
        import nninteractive_bridge as bridge
        with tempfile.TemporaryDirectory() as temp:
            def initialize(context, _payload):
                context.model_dir=str(Path(temp)/'model')
                context.runtime_work_dir=temp
                context._acquire_image_lease()
            with mock.patch.object(bridge._BridgeSessionContext,'_initialize',initialize):
                first=bridge._BridgeSessionContext({})
                with self.assertRaisesRegex(RuntimeError,'另一个 nnInteractive'):
                    bridge._BridgeSessionContext({})
                self.assertTrue(first._image_lease.path.exists())
                first._release_image_lease()
            session=mock.Mock()
            def failing(context,payload):
                initialize(context,payload);context.session=session
                raise ValueError('invalid model metadata')
            with mock.patch.object(bridge._BridgeSessionContext,'_initialize',failing):
                with self.assertRaisesRegex(ValueError,'invalid model'):
                    bridge._BridgeSessionContext({})
            session.close.assert_called_once()
            with mock.patch.object(bridge._BridgeSessionContext,'_initialize',initialize):
                next_context=bridge._BridgeSessionContext({})
                next_context._release_image_lease()

    def test_reaper_rechecks_process_creation_identity_before_kill(self):
        common=runtime.runtime_common
        class InlineThread:
            def __init__(self,target,**kwargs):self.target=target
            def start(self):self.target()
        process=mock.Mock(pid=1234567);process.poll.return_value=None
        with mock.patch.object(common.threading,'Thread',InlineThread), \
             mock.patch.object(common,'process_exists',return_value=True), \
             mock.patch.object(common,'process_start_marker',side_effect=['owned','owned','recycled','recycled']), \
             mock.patch.object(common.os,'kill') as kill, \
             mock.patch.object(common.subprocess,'Popen') as taskkill:
            common.terminate_process_async(process=process,expected_start_marker='owned',graceful_seconds=0)
        process.terminate.assert_called_once()
        kill.assert_not_called();taskkill.assert_not_called()


class PhysicalCache(unittest.TestCase):
    def test_physical_semantics_invalidate_real_materialized_case(self):
        import hashlib
        import nibabel as nib
        import numpy as np
        import nnunet_pipeline as pipeline
        import nnunet_common as common
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'image.raw'
            raw.write_bytes(np.arange(8,dtype=np.int16).tobytes())
            label=root/'label.nii.gz';nib.save(nib.Nifti1Image(np.ones((2,2,2),dtype=np.uint8),np.eye(4)),label)
            request=common.normalize_request(dict(operation='train',task_name='fixture',workspace=str(root/'workspace'),labels=[dict(name='target',id=1)]))
            payload=dict(path=str(raw),shape=[2,2,2],dtype='int16',sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                         mimics_voxel_to_ras_matrix=np.eye(4).tolist(),source_intensity_recovery_basis='mimics_hu_gv',buffer_to_source_slope=1,
                         expected_grid=dict(source_shape=[2,2,2],source_voxel_to_ras_matrix=np.eye(4).tolist()))
            def prepare(p):return pipeline._materialize_case('case',raw,[[label]],request,root/'cache',image_payload=p)
            self.assertFalse(prepare(payload)['cache_hit'])
            self.assertTrue(prepare(payload)['cache_hit'])
            changed=dict(payload,buffer_to_source_slope=2)
            second=prepare(changed);self.assertFalse(second['cache_hit'])
            np.testing.assert_array_equal(np.asarray(nib.load(second['image']).dataobj).ravel(),np.arange(8)*2)
            altered=dict(changed,buffer_to_source_zero_tolerance=3)
            self.assertFalse(prepare(altered)['cache_hit'])
            raw.write_bytes(b'\x00'*16)
            with self.assertRaisesRegex(RuntimeError,'校验失败'):prepare(altered)

    def test_normal_receipt_name_and_real_archive_are_distinguished(self):
        import create_mcs_batch as creator
        import import_undo_mimics as undo
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case='kidney.previous-0123456789abcdef'
            project=root/(case+'.mcs');project.write_bytes(b'fixture')
            payload=creator._import_receipt_payload(case,str(project),[],{},mask_objects=[])
            receipt=Path(creator._receipt_path_for(str(project),case));receipt.write_text(json.dumps(payload))
            archive=receipt.with_name(receipt.name[:-len(undo.RECEIPT_SUFFIX)]+'.previous-abcdef1234567890'+undo.RECEIPT_SUFFIX)
            archived=dict(payload,created_at_epoch=payload['created_at_epoch']+100);archive.write_text(json.dumps(archived))
            with mock.patch.object(undo,'_candidate_receipt_dirs',return_value=[temp]):
                found,_=undo.find_latest_receipt()
            self.assertEqual(str(receipt),found)


if __name__=='__main__':
    unittest.main(verbosity=2)
