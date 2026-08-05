# Remote AI Server Setup

The remote runtime is optional. Local DINOv3, nnInteractive, and nnU-Net
training or inference do not use it unless a user explicitly chooses a saved
remote server in the corresponding external window.

## Build directly on the GPU server

Clone or copy the project to the Linux server, install Docker and NVIDIA
Container Toolkit, then run:

```bash
MIMICS_AI_ROOT=/srv/mimics-ai \
  remote/setup_remote_server.sh --build
```

This builds `mimics-ai-runtime:1.0`, creates the work folders, checks the
installed base weights, all three frameworks, and the offline container
contract. The image is built
once. Starting a later training job creates a small disposable container from
the existing image and does not rebuild the image.

The image contains a snapshot of the project code. After pulling changes that
touch `tools/`, `runtime_py35/`, or an `external/` training package, rebuild the
image with `remote/setup_remote_server.sh --build` before starting another
remote job. Dataset caches and model weights live outside the image and are not
deleted by this rebuild.

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
/srv/mimics-ai/models/dinov3/dinov3-vits16/model.onnx
/srv/mimics-ai/models/nninteractive/nnInteractive_v1.0/
```

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
- Cached datasets expire after 30 days without use.
- `remote_worker.log` rotates at 32 MiB with three backups.
- Before downloading a model or nnU-Net prediction, the Windows controller
  checks space for the remaining transfer, unpacked artifact, and a safety
  margin. Existing registered models are not replaced on a failed download.
- If SSH remains unavailable, Stop never claims the container was stopped. The
  status window offers **Abandon Locally** as an explicit last resort and keeps
  the server/container identity for administrator cleanup.
