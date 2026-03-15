# Optimized package namespace (no pollution)

To install the optimized variant under a separate Python module name,
use `setup_optimized.py` instead of `setup.py`.

## Install optimized variant

```bash
cd submodules/diff-gaussian-rasterization
python setup_optimized.py develop
```

This installs a separate package:

- `diff_gaussian_rasterization_wfeat_optimized`

with a separate extension module:

- `diff_gaussian_rasterization_wfeat_optimized._C`

So it does not overwrite/import-collide with:

- `diff_gaussian_rasterization`
- `diff_gaussian_rasterization_wfeat`

## Usage example

```python
import diff_gaussian_rasterization_wfeat_optimized as dgr_opt

ssr = dgr_opt.Gaussian_SSR(
    tanfovx, tanfovy, W, H,
    radius, bias, thick, delta, step, start,
    backward_mode="fast_backward",  # or "full_backward"
)

color, abd = ssr(normal, pos, rgb, albedo, roughness, metallic, F0)
```
