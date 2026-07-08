# Syll sandbox — Docker backend

`DockerEnvironment` (`syll/sandbox/backends/docker.py`) runs primitives inside a
Docker container via `docker exec` and gives ENPIRE-style snapshot/reset:

- `reset("full")` — remove + recreate the container from the base image
  (deterministic clean slate).
- `checkpoint(tag)` — `docker commit` the running container to a tagged image.
- `restore(tag)` / `reset("phase", checkpoint_id)` — recreate from a tagged
  image (ENPIRE phase-reset; MACU `init_from` / `variant_of`).

## Build the base image

Minimal image (enough for reset-determinism + file/shell tasks):

```bash
docker build -t syll-sandbox-base:latest -f syll/sandbox/docker/Dockerfile syll/sandbox/docker
```

Desktop image (real GUI tasks — screenshot / xdotool / browser / LibreOffice):
uncomment the desktop variant block at the bottom of the `Dockerfile` and build
it as a separate tag, e.g. `syll-sandbox-desktop:latest`. It adds XFCE + Xvfb +
x11vnc + xdotool + ImageMagick + Firefox + LibreOffice. Start Xvfb on `:1`
(the default `run_args` passes `-e DISPLAY=:1`).

## Verify reset determinism

With Docker running and the base image built:

```bash
pytest tests/test_docker_reset_determinism.py -q
```

The test skips automatically when Docker or the image is absent.

## Run the sandbox server

```bash
python -m syll.sandbox.server --backend docker --image syll-sandbox-base:latest --port 8086
# then drive it over the HTTP wire protocol (syll/sandbox/protocol.py)
```
