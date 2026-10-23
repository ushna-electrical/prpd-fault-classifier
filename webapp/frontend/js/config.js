/* Deployment config -- the one file to edit per environment. Loaded before
 * every other script on every page.
 *
 * Local development (two separate servers, the default this project ships
 * with -- a static file server for this frontend, FastAPI on port 8000):
 *     API_BASE_URL: "http://127.0.0.1:8000"
 *
 * Single-process deployment (FastAPI also serves this frontend, e.g. with
 * PRPD_SERVE_FRONTEND=1 behind one Nginx reverse proxy, or Nginx proxying
 * "/api/" to FastAPI and serving these static files itself) -- same origin,
 * so just use a relative path:
 *     API_BASE_URL: ""
 *
 * Split deployment (this frontend on a static host like Vercel, the backend
 * on a separate server, e.g. a Hostinger VPS) -- point at the backend's own
 * public URL. The backend's PRPD_CORS_ORIGINS must then include this
 * frontend's origin:
 *     API_BASE_URL: "https://api.yourdomain.com"
 *
 * Every backend route lives under /api (see app/main.py), so this value is
 * always just the origin/prefix the frontend prepends to paths like
 * "/api/predict" -- never include "/api" here.
 */
window.PRPD_CONFIG = {
  API_BASE_URL: ""
};
