# Minimal Syll sandbox base image.
#
# Enough to verify DockerEnvironment snapshot/reset semantics (exec, file I/O,
# full reset, phase checkpoint/restore). Small and fast to build. For real GUI
# tasks build the desktop variant below instead.
#
#   docker build -t syll-sandbox-base:latest -f syll/sandbox/docker/Dockerfile syll/sandbox/docker

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 ca-certificates coreutils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/workspace
WORKDIR /root/workspace

# Kept alive by `tail -f /dev/null` at run time (see DockerEnvironment).
CMD ["tail", "-f", "/dev/null"]

# -------------------------------------------------------------------------
# Desktop variant (uncomment / build separately for real GUI tasks):
#
# FROM ubuntu:22.04
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       xfce4 xvfb x11vnc python3 ca-certificates \
#       xdotool imagemagick scrot \
#       firefox libreoffice \
#     && rm -rf /var/lib/apt/lists/*
# # Xvfb on :1, XFCE, x11vnc on 5900 — start via an entrypoint script.
# -------------------------------------------------------------------------
