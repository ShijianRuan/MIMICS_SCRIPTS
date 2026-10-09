# Remote AI Server Setup

The remote runtime is optional. Local nnInteractive and nnU-Net
training or inference do not use it unless a user explicitly chooses a saved
remote server in the corresponding external window.

## Updating the runtime image (standard flow, no local Docker)

The image contains a snapshot of the project code. After changes that touch
`tools/`, `runtime_py35/`, or an `integrations/` training package, update the
image with one command from the Windows workstation — the build runs **on** the
GPU server over SSH, so the workstation needs no Docker at all:

```bash
python_env/python.exe tools/remote_training_controller.py build-image \
  --profile-id <saved-profile-id>
```

What the command does, and what to check at each step:

1. **Trigger** — run it whenever the remote controller's code-drift check
   rejects a job (the status window reports that the image is older than the
   local code), or after pulling changes that touch the shipped code.
2. **Upload** — the checkout is packed code-only (local caches, the 12.8G
   model registries, and weight folders stay out; ~14 MB). A digest marker
   skips re-uploading unchanged code on a retry.
3. **Build** — `nerdctl/docker build` runs on the server into the profile's
   container namespace. A missing nerdctl/docker on the server is detected
   before any upload and refuses with a message naming
   `remote/setup_remote_server.sh`. Base-image pulls go through the
   server's network; if Docker Hub or your usual registry is unreachable,
   pass a mirror via `--base-image`. Verified options (2026-10, from the
   206 workstation's network):
   - `registry.united-imaging.com/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
     (internal UI mirror — confirmed reachable from 206 and carrying the
     exact base tag),
   - or a digest-pinned reference to a layer already on the server
     (e.g. `…@sha256:77f17f84…`), which skips registry metadata lookups
     entirely.
   The pip layer cache makes a rebuild after a code-only change take ~15 min.
4. **Verify** — after the build, confirm two things:
   - the A1 image preflight passes for every job kind (the controller runs
     this automatically before each job; a fresh build must pass all five), and
   - the next remote job's code-fingerprint check reports a match (it runs
     automatically at job submit).
   Both checks are automatic — an image that fails either refuses jobs
   instead of silently training with stale code.
5. **Rollback** — images are tagged; keep the previous image ID (shown by the
   build command, also in the server's `logs/build_*.log`). If a fresh image
   misbehaves, retag the previous ID back:

   ```bash
   nerdctl -n <namespace> tag <old-image-id> mimics-ai-runtime:1.0
   ```

   Dataset caches and model weights live outside the image and survive a
   rebuild; the old image blobs are garbage-collected by the server's normal
   containerd pruning.

## First-time setup: build directly on the GPU server

Clone or copy the project to the Linux server, install Docker and NVIDIA
Container Toolkit, then run:

```bash
MIMICS_AI_ROOT=/srv/mimics-ai \
  remote/setup_remote_server.sh --build
```

This builds `mimics-ai-runtime:1.0`, creates the work folders, checks the
installed base weights, all supported frameworks (nnInteractive, nnU-Net,
FlexiCT), and the offline container contract. The image is built
once. Starting a later training job creates a small disposable container from
the existing image and does not rebuild the image. After that, use the
`build-image` flow above for every code update — no server-side checkout or
manual build is needed.

## Build once and transfer an image archive

On a Linux build machine:

```bash
remote/build_image.sh /tmp/mimics-ai-runtime.tar
```

Copy the archive and project checkout to the GPU server, then run:

```bash
MIMICS_AI_ROOT=/srv/mimics-ai \
  remote/setup_remote_server.sh /tmp/mimics-ai-runtime.tar
```

This route is suitable for a server that cannot download Python packages while
building.

## Required model layout

Before setup completes, install the base weights at:

```text
/srv/mimics-ai/models/nninteractive/nnInteractive_v1.0/
/srv/mimics-ai/models/flexict/flexict_2d/model.safetensors
/srv/mimics-ai/models/flexict/flexict_3d/model.safetensors
```

The FlexiCT backbone pair initializes every remote FlexiCT fine-tune; its
content fingerprint is checked against the local backbone weights before a
training job uploads data.

Use the same directory as the `Remote work folder` in the Windows client
profile.

## Runtime lifecycle

- The client connects to the Linux host over SSH. Training containers do not
  run SSH and do not expose a network port.
- Containers have `--network none`; Hugging Face, Transformers, datasets, and
  experiment tracking are forced offline. Required weights must exist under
  the read-only `models` mount before a job starts.
- One training job uses one GPU and one disposable container.
- `Automatic` GPU selection serializes against all explicitly selected GPUs.
- Selecting GPU `0`, GPU `1`, or a GPU UUID creates an independent queue for
  that device.
- Successful and cancelled containers are removed after their terminal state
  is confirmed.
- Failed containers are stopped and removed, while their job files and logs
  remain for diagnosis.
- Remote job directories are not deleted by age alone; this avoids deleting a
  long-running task whose parent-directory timestamp has not changed.
- Every stop and cleanup verifies the container owner/job labels and confines
  deletion to `jobs/<ssh-user>/<job-id>`.
- The Docker image and base weights remain installed.
- Unchanged image and label archives are reused per case by content
  fingerprint, so adding one case does not re-upload the full dataset.
- nnU-Net raw and preprocessed data are isolated by the complete dataset
  fingerprint. A changed case creates a new cache namespace instead of
  contaminating an earlier plan.
- nnU-Net models are downloaded into the same portable local registry used by
  local training. Remote inference reuses model and image transfer archives by
  content identity.
- FlexiCT models download as one folder per trained configuration (a pair
  job yields 2D + 3D sharing a pair_id) and register into the existing local
  FlexiCT registry with full remote provenance.
- Cached datasets expire after 30 days without use.
- `remote_worker.log` rotates at 32 MiB with three backups.
- Before downloading a model or nnU-Net prediction, the Windows controller
  checks space for the remaining transfer, unpacked artifact, and a safety
  margin. Existing registered models are not replaced on a failed download.
- If SSH remains unavailable, Stop never claims the container was stopped. The
  status window offers **Abandon Locally** as an explicit last resort and keeps
  the server/container identity for administrator cleanup.
