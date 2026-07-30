# `deploy/models/` — default policy-artifact mount

Bind-mounted **read-only** at `/models` in the `inference` service
(`docker-compose.yml`: `${STRAFER_MODELS_DIR:-./models}:/models:ro`).

This directory ships empty and exists only so the default bind source is a real
path in a fresh checkout. With no default directory behind it,
`STRAFER_MODELS_DIR` unset rendered a bind source that did not exist — while the
artifact under test sat one directory away.

Put the exported policy here:

```
deploy/models/
├── policy.onnx          # artifact
└── policy.json          # sidecar (variant, obs_dim, git_commit)
```

…or leave it empty and point elsewhere, which is what the NX does:

```bash
# deploy/.env  (cp .env.example .env)
STRAFER_MODELS_DIR=/home/<user>/strafer_models
```

Select **which** artifact under `/models` is loaded with
`STRAFER_INFERENCE_MODEL_PATH` (host env or `deploy/.env`) — not by editing a
tracked compose file. Artifacts are large binaries and are **not** committed.

An empty or missing model under a policy backend aborts the inference container
at launch by design (`inference_policy.launch.py`) — no silent nav2 fallback.
