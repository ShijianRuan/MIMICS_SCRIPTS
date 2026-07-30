# Remote Training Server Setup

The remote runtime is an optional addition. Local DINOv3 and nnInteractive
training do not use these files unless a user explicitly chooses a saved
remote server in the training window.

## Build directly on the GPU server

Clone or copy the project to the Linux server, install Docker and NVIDIA
Container Toolkit, then run:

```bash
MIMICS_AI_ROOT=/srv/mimics-ai \
  remote/setup_remote_server.sh --build
```

This builds `mimics-ai-runtime:1.0`, creates the work folders, checks the
installed base weights, and runs the container preflight. The image is built
once. Starting a later training job creates a small disposable container from
the existing image and does not rebuild the image.

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
- Cached datasets expire after 30 days without use.
