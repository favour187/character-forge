# Neural 3D model

Character Forge can run a real learned image-to-3D model locally through
neural3d.py.

## Backend

The default neural backend is TripoSR. Its code and pretrained model are
MIT-licensed. The first neural build downloads the pretrained weights from
Hugging Face into the local cache; mesh generation then runs inside the
Character Forge process.

This is intentionally separate from requirements.txt: the normal web app
still fits lightweight CPU deployments, while the neural engine requires a
machine with enough memory/VRAM.

## Enable

Install:

pip install -r requirements-neural3d.txt

Then set:

FORGE_3D_ENGINE=triposr

Useful controls:

TRIPOSR_DEVICE=cuda:0
TRIPOSR_MC_RESOLUTION=256
TRIPOSR_CHUNK_SIZE=8192

If the neural dependencies are not installed, Character Forge keeps its
existing CPU reconstruction path rather than crashing.

## Important

Do not commit the pretrained checkpoint into normal Git history. It is a large
binary model artifact. Keeping the loader/configuration in the repository and
caching the weights locally is the portable approach.
