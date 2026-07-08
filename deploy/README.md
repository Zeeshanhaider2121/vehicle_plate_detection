# PlateFlow — EC2 GPU Deployment (Docker + GitHub Actions)

One container runs the whole app: `uvicorn app.main:app` on **:8000** loads YOLO on
the GPU, runs the PaddleOCR worker (separate venv), serves the REST API **and** the
built React frontend.

```
git push ─▶ GitHub Actions (cloud) ─▶ docker build ─▶ push ghcr.io/<you>/<repo>
                                          └▶ scp compose + ssh ec2 ─▶ compose pull && up -d
EC2 (GPU DLAMI):  container :8000   volumes: /opt/plateflow/{models(ro),data,outputs}
```

Files in this repo: `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
`.github/workflows/deploy.yml`, `deploy/ec2-setup.sh`, `deploy/.env.production.example`.

---

## 0. Launch the EC2 instance
- AMI: the GPU DLAMI you have (Amazon Linux 2023, driver 595 / CUDA 13.x).
- Instance type: **g5.xlarge** or **g6.xlarge** (1× GPU, 24 GB VRAM) is plenty.
- Storage: **≥ 100 GB** gp3 (the image + models + CUDA layers are big).
- Security group inbound: **TCP 22** (SSH, your IP) and **TCP 8000** (the app).

## 1. Bootstrap the host (once)
```bash
scp -i key.pem deploy/ec2-setup.sh ec2-user@<EC2_IP>:~
ssh -i key.pem ec2-user@<EC2_IP> 'bash ec2-setup.sh'
# then log out and back in so `docker` works without sudo
```
This installs Docker + Compose v2 + the NVIDIA Container Toolkit and creates
`/opt/plateflow/{models,data,outputs}`. It ends with a `nvidia-smi`-in-Docker test.

## 2. Upload the model weights (once, ~1.1 GB)
They're gitignored, so they never ride in the image — they live on the host:
```bash
scp -i key.pem backend/model/V4.pt backend/model/best_V2.pt \
    ec2-user@<EC2_IP>:/opt/plateflow/models/
```
(Add any other `.pt` files your `.env` references. Re-scp only when models change.)

## 3. Create the host env file (once)
```bash
scp -i key.pem deploy/.env.production.example ec2-user@<EC2_IP>:/opt/plateflow/.env
ssh -i key.pem ec2-user@<EC2_IP> 'nano /opt/plateflow/.env'
```
Set at minimum:
- `IMAGE=ghcr.io/<owner>/<repo>:latest`  ← **lowercase** owner/repo
- `MINERU_TOKEN=<your token>`

The `docker-compose.yml` is copied to `/opt/plateflow/` automatically by the deploy
job, so you don't place it by hand. `.env` is the only host file you manage.

## 4. Add GitHub repository secrets
`Settings → Secrets and variables → Actions → New repository secret`:

| Secret | Value |
|--------|-------|
| `EC2_HOST` | instance public IP or DNS |
| `EC2_USER` | `ec2-user` |
| `EC2_SSH_KEY` | contents of your `.pem` private key |
| `GHCR_PAT` | a **classic PAT** with `read:packages` (used by EC2 to pull the private image) |

Pushing the image uses the built-in `GITHUB_TOKEN` (the workflow already grants
`packages: write`) — no secret needed for that. `GHCR_PAT` is only for the *pull*
on EC2. (Alternatively make the GHCR package public and skip `GHCR_PAT`.)

## 5. First deploy
Push to `main` (or `demo_v0`), or run the workflow manually
(`Actions → Build & Deploy PlateFlow → Run workflow`). It builds, pushes, then
deploys over SSH. First container boot loads the 565 MB YOLO model — give it 1–3 min.

Watch it come up:
```bash
ssh -i key.pem ec2-user@<EC2_IP>
cd /opt/plateflow && docker compose logs -f
#  [PlateFlow] GPU enabled: CUDA (NVIDIA ...)
#  [PlateFlow] GPU OCR worker ready at http://127.0.0.1:8899
```

## 6. Verify
```bash
curl http://<EC2_IP>:8000/health      # -> {"status":"ok",...}
```
Open **http://<EC2_IP>:8000** in a browser.

---

## Updating
- **Code/frontend:** just `git push` → Actions rebuilds and redeploys.
- **Models:** re-`scp` to `/opt/plateflow/models/` (no rebuild needed; restart with
  `docker compose restart` if the app was already running).
- **Config:** edit `/opt/plateflow/.env` on the host, then `docker compose up -d`.

## Data & backups
Everything that must survive redeploys lives under `/opt/plateflow`:
`data/plateflow.db` (SQLite) and `outputs/` (crops, logs, media). Back these up.

## Troubleshooting
- **`docker: could not select device driver ... gpu`** → toolkit not configured; re-run
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
- **Compose rejects `gpus: all`** → delete that line and uncomment the `deploy:` block
  at the bottom of `docker-compose.yml`.
- **OCR fields stay `'—'`** → check `docker compose logs` for the OCR worker. Confirm
  `OCR_GPU_WORKER_PYTHON=/opt/venv-ocr/bin/python` in `.env` and a valid `MINERU_TOKEN`.
- **`paddlepaddle-gpu` failed to build** → in the `Dockerfile` OCR-venv step, swap
  `paddlepaddle-gpu` → `paddlepaddle` (and drop the `.cn` index + the `nvidia-cudnn`
  line). OCR then runs on CPU — slower but reliable.
- **Runner runs out of disk while building** → the `Free up disk space` step handles it;
  if it still fails, set `large-packages: true` in that step.
- **Image pull denied on EC2** → the GHCR package is private; ensure `GHCR_PAT` has
  `read:packages`, or make the package public in your GitHub packages settings.
