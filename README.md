# KaRMA project page

Static project page for **"A Kinematic Metric for Fine Manipulation Ability in
Robotic Hands"** (KaRMA, IROS 2026, arXiv:2605.15548). Meant to be served at
`martinpeticco.com/karma/`.

No build step. Plain HTML + CSS + a small ES-module three.js viewer.

## Structure

```
index.html                     the whole page (SEO head, sections, inline leaderboard JSON + scripts)
static/css/index.css           theme
static/js/karma_viewer.js       three.js workspace explorer (ES module)
static/js/vendor/               self-hosted three.js 0.160.0 + OrbitControls
static/scenes/scenes.json       per-hand geometry for all 16 hands (~100 KB)
static/images/                  metric_overview.png, metric_method.png, examples.png,
                                social_preview.png (1200x630 og:image)
tools/export_web_scenes.py      regenerates scenes.json from the code repo's results
.nojekyll
```

The interactive explorer draws each hand's seed-grasp scene (capsule links, the
test sphere, contact normals) plus the reachable voxel cloud colored by rotation
coverage on a fixed red->green scale, matching the paper figures. All geometry is
procedural, so no third-party hand meshes are shipped.

## Regenerate the scene data

Run from the code repo (`karma-hand-metric`) in its conda env:

```bash
PYTHONPATH=. conda run -n karma-hand-metric --cwd /path/to/karma-hand-metric \
  python /path/to/karmasite/tools/export_web_scenes.py \
  --out /path/to/karmasite/static/scenes/scenes.json
```

The leaderboard numbers in `index.html` (inline `#lb-data`) are Table I of the
paper; the ablation mini-table is Table IV.

## Preview locally

```bash
cd karmasite && python -m http.server 8899
# open http://127.0.0.1:8899/
```

## Deploying (not yet wired)

The page is built but not published. To serve it at `martinpeticco.com/karma/`:

1. Create a **public GitHub repo named exactly `karma`** under `mfpeticco` and push
   this directory to it. The repo name is the URL path — GitHub Pages serves a
   user's project repo at `<apex-domain>/<repo-name>/`, so it must be `karma`, not
   `karmasite`. (The local folder name is free to differ, same as `dexwristsite` ->
   the `dexwrist` repo.)
2. Enable Pages: **branch `master` (or `main`), path `/` (root)**. The `.nojekyll`
   file is already present. No CNAME is needed in this repo; the apex domain is
   owned by the `mfpeticco.github.io` user site.
3. Remove the old redirect stub `karma/index.html` from the `mfpeticco.github.io`
   repo so the project page takes over `/karma`.
4. Add `website = {https://martinpeticco.com/karma/}` to the `peticco2026karma`
   entry in `mfpeticco.github.io`'s `_bibliography/papers.bib` (matches the DexWrist
   entry).
5. After it's live, validate the social card with the Twitter/Facebook debuggers.
```
