#!/usr/bin/env bash
# One-time EC2 host bootstrap for PlateFlow on Amazon Linux 2023 (GPU DLAMI).
# Installs Docker + Compose v2 + NVIDIA Container Toolkit and creates the data dirs.
# The DLAMI may already have some of these — every step is idempotent.
#
#   scp deploy/ec2-setup.sh ec2-user@<host>:~   &&   ssh ec2-user@<host> 'bash ec2-setup.sh'
set -euo pipefail

echo "==> Docker engine"
if ! command -v docker >/dev/null 2>&1; then
  sudo dnf install -y docker
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true   # take effect on next login

echo "==> Docker Compose v2 (CLI plugin)"
PLUGIN_DIR=/usr/libexec/docker/cli-plugins
if ! docker compose version >/dev/null 2>&1; then
  sudo mkdir -p "$PLUGIN_DIR"
  sudo curl -fsSL \
    "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
    -o "$PLUGIN_DIR/docker-compose"
  sudo chmod +x "$PLUGIN_DIR/docker-compose"
fi

echo "==> NVIDIA Container Toolkit (lets Docker use the GPU)"
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
  sudo dnf install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "==> App directories (models mounted read-only; data + outputs persist)"
sudo mkdir -p /opt/plateflow/models /opt/plateflow/data /opt/plateflow/outputs
sudo chown -R "$USER":"$USER" /opt/plateflow

echo "==> GPU-in-Docker smoke test"
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi \
  || echo "!! GPU test failed — check the NVIDIA driver / toolkit before deploying."

echo
echo "Done. Next:"
echo "  1) Log out & back in (so 'docker' works without sudo)."
echo "  2) Upload model weights to /opt/plateflow/models/  (V4.pt, best_V2.pt, ...)."
echo "  3) Put docker-compose.yml + .env in /opt/plateflow/ (see deploy/README.md)."
