# Deployment

Three ways to run this outside local development, from simplest to most capable.
All of them need the backend (FastAPI + PyTorch) running somewhere with Python and
the exported model weights — that part cannot run on Vercel (see "Why not Vercel
for the backend" below).

## 0. What changed to make this possible

- Every backend route now lives under `/api` (`app/main.py`), so one reverse proxy
  rule (`/api/* → the backend`) is enough regardless of topology.
- The backend reads its deployment-specific settings from environment variables,
  all optional (unset = exactly the original local-dev behavior):

  | Variable | Purpose | Default |
  |---|---|---|
  | `PRPD_CORS_ORIGINS` | comma-separated allowed browser origins | `*` |
  | `PRPD_LIBRARY_IMAGES_DIR` | Fault Library example images folder | `../Table 5.5` (next to `Code/`) |
  | `PRPD_STAGE2_DIR` | dashboard source files folder | `Code/stage2_output` |
  | `PRPD_SERVE_FRONTEND` | `1` to also serve the frontend from this same process | unset (off) |
  | `PRPD_FRONTEND_DIR` | frontend folder, if `PRPD_SERVE_FRONTEND=1` | `Code/webapp/frontend` |

- The frontend reads the backend's origin from **one file**, `frontend/js/config.js`
  (`window.PRPD_CONFIG.API_BASE_URL`). That's the only edit needed per environment.
- `frontend/vercel.json` makes the `frontend/` folder deployable to Vercel as-is.

## 1. Single VPS (recommended for this project)

Everything — backend and frontend — on one Hostinger VPS (or any VPS/droplet).

1. `git clone`/upload the repo to the VPS, e.g. to `/opt/prpd/Code` (same relative
   layout as your machine: the `Dataset/`, `Table 5.5/` and `Code/` folders stay
   siblings, since the backend defaults assume that).
2. Create the venv and install backend deps:
   ```bash
   cd /opt/prpd/Code
   python3 -m venv .venv
   .venv/bin/pip install -r webapp/backend/requirements.txt
   ```
3. Generate the dashboard's confusion-matrix files once (see `webapp/README.md`
   step 2) if they're not already present under `stage2_output/`.
4. Copy `webapp/deploy/prpd-backend.service` to `/etc/systemd/system/`, adjust its
   paths/user, copy `webapp/deploy/prpd-backend.env.example` to
   `/opt/prpd/prpd-backend.env` (leave everything commented out — same-origin
   doesn't need `PRPD_CORS_ORIGINS`), then:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now prpd-backend
   sudo systemctl status prpd-backend   # should say "active (running)"
   ```
5. Point your subdomain's DNS **A record** at the VPS's IP.
6. Install Nginx + certbot, copy `webapp/deploy/nginx-prpd.conf.example` to
   `/etc/nginx/sites-available/prpd.conf` (edit the domain/paths), enable it, then
   run `sudo certbot --nginx -d prpd.yourdomain.com` for the TLS cert.
7. In `frontend/js/config.js`, set `API_BASE_URL: ""` (same-origin via Nginx).
8. Visit `https://prpd.yourdomain.com` — done.

Decide now whether this subdomain should be public. It's exposing lab
measurements, trained models and thesis results that may not be published yet —
the Nginx config's `auth_basic` lines (commented out) add a username/password
prompt in front of everything if you want that.

## 2. Split: frontend on Vercel, backend on a VPS

If you'd rather use Vercel for the frontend specifically (instant global CDN,
zero server maintenance for the static part):

1. Do steps 1–6 above for the **backend only** (skip step 7; you don't need Nginx
   to serve the frontend in this setup, only to reverse-proxy `/api/` — or just
   point the subdomain straight at the backend with `PRPD_SERVE_FRONTEND` unset
   and a simpler Nginx config that only has the `location /api/` block).
2. In Vercel: **New Project → Import** this repo, set the project's **Root
   Directory** to `webapp/frontend`. Framework preset: "Other" (it's static
   HTML/CSS/JS, no build step, no build command needed).
3. Before deploying, edit `frontend/js/config.js`:
   ```js
   window.PRPD_CONFIG = { API_BASE_URL: "https://prpd-api.yourdomain.com" };
   ```
4. On the backend, set `PRPD_CORS_ORIGINS` in `/opt/prpd/prpd-backend.env` to your
   Vercel URL(s) (the `*.vercel.app` preview domain and/or your custom domain),
   then `sudo systemctl restart prpd-backend`.
5. Deploy on Vercel. Visit the Vercel URL — it now calls your VPS backend directly.

Same privacy note as above: the backend subdomain is now a separate, public
origin unless you add `auth_basic` (or similar) in front of it.

## 3. Local development (unchanged)

Still the default, two servers, documented in `webapp/README.md`. `config.js`
ships with `API_BASE_URL: "http://127.0.0.1:8000"`, matching that setup exactly —
no changes needed to develop locally after this update.

## Why not Vercel for the backend

Vercel's Python serverless functions cap out around 250MB uncompressed —
`torch`+`torchvision` alone are typically close to or over that before adding the
~135MB of model weights. Vercel functions also can't see files outside the
deployed project (no access to sibling `Dataset/`/`Table 5.5/`/`stage2_output/`
folders), and cold starts would mean reloading the ensemble on every idle-then-hit
request. None of that fits a persistent PyTorch service — hence the split above
instead of trying to force the backend onto Vercel too.

## Troubleshooting

- **CORS error in the browser console (split deployment only)**: `PRPD_CORS_ORIGINS`
  on the backend doesn't include the frontend's exact origin (scheme + host, no
  trailing slash). Restart the backend after changing it.
- **413 on upload**: Nginx's `client_max_body_size` is below the backend's 15 MB
  limit, or vice versa — they should match (see the Nginx example).
- **Images/CSVs 404 on the dashboard or library**: `PRPD_LIBRARY_IMAGES_DIR` /
  `PRPD_STAGE2_DIR` don't point at real folders on that machine, or those folders
  weren't uploaded.
- **Backend works via `curl` but not the browser**: check `frontend/js/config.js`
  matches how you actually deployed (same-origin `""` vs. the backend's real URL).
