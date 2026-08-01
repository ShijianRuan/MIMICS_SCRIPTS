# ScribblePrompt Runtime

This directory contains the official ScribblePrompt UNet inference architecture
from commit `182c44975f77749b559974ce8db558c8bde57788`. It is used under the
upstream Apache-2.0 license.

Place the official UNet checkpoint at:

`external/ScribblePrompt/checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt`

Official checkpoint URL:

`https://www.dropbox.com/scl/fi/pnw88n05irnv5z1snlklr/ScribblePrompt_unet_v1_nf192_res128.pt?rlkey=dr8xvkf0wj2r082h1zzpcmz5o&dl=1`

On an internet-connected packaging workstation, run:

`python tools/download_scribbleprompt_checkpoint.py`

The checkpoint is intentionally not committed to Git. Portable packages include
it automatically when it is present locally.
