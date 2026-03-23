# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import os
import sys
import torch
import torch.utils.cpp_extension

from .bsdf import *
from .loss import *

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None
_plugin_load_failed = False
def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin, _plugin_load_failed
    if _cached_plugin is not None:
        return _cached_plugin
    if _plugin_load_failed:
        return None

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                print("Warning: Could not locate a supported Microsoft Visual C++ installation, falling back to PyTorch implementations")
                _plugin_load_failed = True
                return None
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/mesh.cu',
        'c_src/loss.cu',
        'c_src/bsdf.cu',
        'c_src/normal.cu',
        'c_src/cubemap.cu',
        'c_src/common.cpp',
        'c_src/torch_bindings.cpp'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False), 'lock')
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    try:
        torch.utils.cpp_extension.load(name='renderutils_plugin', sources=source_paths, extra_cflags=opts,
             extra_cuda_cflags=opts, extra_ldflags=ldflags, with_cuda=True, verbose=True)

        # Import, cache, and return the compiled module.
        import renderutils_plugin
        _cached_plugin = renderutils_plugin
        return _cached_plugin
    except Exception as e:
        print(f"Warning: Failed to compile renderutils_plugin: {e}")
        print("Falling back to PyTorch implementations for cubemap filtering.")
        _plugin_load_failed = True
        return None

#----------------------------------------------------------------------------
# Internal kernels, just used for testing functionality

class _fresnel_shlick_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f0, f90, cosTheta):
        out = _get_plugin().fresnel_shlick_fwd(f0, f90, cosTheta, False)
        ctx.save_for_backward(f0, f90, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        f0, f90, cosTheta = ctx.saved_variables
        return _get_plugin().fresnel_shlick_bwd(f0, f90, cosTheta, dout) + (None,)

def _fresnel_shlick(f0, f90, cosTheta, use_python=False):
    if use_python:
        out = bsdf_fresnel_shlick(f0, f90, cosTheta)
    else:
        out = _fresnel_shlick_func.apply(f0, f90, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _fresnel_shlick contains inf or NaN"
    return out


class _ndf_ggx_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosTheta):
        out = _get_plugin().ndf_ggx_fwd(alphaSqr, cosTheta, False)
        ctx.save_for_backward(alphaSqr, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosTheta = ctx.saved_variables
        return _get_plugin().ndf_ggx_bwd(alphaSqr, cosTheta, dout) + (None,)

def _ndf_ggx(alphaSqr, cosTheta, use_python=False):
    if use_python:
        out = bsdf_ndf_ggx(alphaSqr, cosTheta)
    else:
        out = _ndf_ggx_func.apply(alphaSqr, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _ndf_ggx contains inf or NaN"
    return out

class _lambda_ggx_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosTheta):
        out = _get_plugin().lambda_ggx_fwd(alphaSqr, cosTheta, False)
        ctx.save_for_backward(alphaSqr, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosTheta = ctx.saved_variables
        return _get_plugin().lambda_ggx_bwd(alphaSqr, cosTheta, dout) + (None,)

def _lambda_ggx(alphaSqr, cosTheta, use_python=False):
    if use_python:
        out = bsdf_lambda_ggx(alphaSqr, cosTheta)
    else:
        out = _lambda_ggx_func.apply(alphaSqr, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _lambda_ggx contains inf or NaN"
    return out

class _masking_smith_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosThetaI, cosThetaO):
        ctx.save_for_backward(alphaSqr, cosThetaI, cosThetaO)
        out = _get_plugin().masking_smith_fwd(alphaSqr, cosThetaI, cosThetaO, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosThetaI, cosThetaO = ctx.saved_variables
        return _get_plugin().masking_smith_bwd(alphaSqr, cosThetaI, cosThetaO, dout) + (None,)

def _masking_smith(alphaSqr, cosThetaI, cosThetaO, use_python=False):
    if use_python:
        out = bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO)
    else:
        out = _masking_smith_func.apply(alphaSqr, cosThetaI, cosThetaO)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _masking_smith contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Shading normal setup (bump mapping + bent normals)

class _prepare_shading_normal_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
        ctx.two_sided_shading, ctx.opengl = two_sided_shading, opengl
        out = _get_plugin().prepare_shading_normal_fwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl, False)
        ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
        return out

    @staticmethod
    def backward(ctx, dout):
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_variables
        return _get_plugin().prepare_shading_normal_bwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, dout, ctx.two_sided_shading, ctx.opengl) + (None, None, None)

def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True, use_python=False):
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
        use_python: Use PyTorch implementation (for validation)
    Returns:
        Final shading normal
    '''    

    if perturbed_nrm is None:
        perturbed_nrm = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    
    if use_python:
        out = bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    else:
        out = _prepare_shading_normal_func.apply(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of prepare_shading_normal contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# BSDF functions

class _lambert_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi):
        out = _get_plugin().lambert_fwd(nrm, wi, False)
        ctx.save_for_backward(nrm, wi)
        return out

    @staticmethod
    def backward(ctx, dout):
        nrm, wi = ctx.saved_variables
        return _get_plugin().lambert_bwd(nrm, wi, dout) + (None,)

def lambert(nrm, wi, use_python=False):
    '''Lambertian bsdf. 
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    if use_python:
        out = bsdf_lambert(nrm, wi)
    else:
        out = _lambert_func.apply(nrm, wi)
 
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out

class _frostbite_diffuse_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi, wo, linearRoughness):
        out = _get_plugin().frostbite_fwd(nrm, wi, wo, linearRoughness, False)
        ctx.save_for_backward(nrm, wi, wo, linearRoughness)
        return out

    @staticmethod
    def backward(ctx, dout):
        nrm, wi, wo, linearRoughness = ctx.saved_variables
        return _get_plugin().frostbite_bwd(nrm, wi, wo, linearRoughness, dout) + (None,)

def frostbite_diffuse(nrm, wi, wo, linearRoughness, use_python=False):
    '''Frostbite, normalized Disney Diffuse bsdf. 
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.
        wo: World space camera vector.
        linearRoughness: Material roughness
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    if use_python:
        out = bsdf_frostbite(nrm, wi, wo, linearRoughness)
    else:
        out = _frostbite_diffuse_func.apply(nrm, wi, wo, linearRoughness)
 
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out

class _pbr_specular_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, col, nrm, wo, wi, alpha, min_roughness):
        ctx.save_for_backward(col, nrm, wo, wi, alpha)
        ctx.min_roughness = min_roughness
        out = _get_plugin().pbr_specular_fwd(col, nrm, wo, wi, alpha, min_roughness, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        col, nrm, wo, wi, alpha = ctx.saved_variables
        return _get_plugin().pbr_specular_bwd(col, nrm, wo, wi, alpha, ctx.min_roughness, dout) + (None, None)

def pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08, use_python=False):
    '''Physically-based specular bsdf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        col: Specular lobe color
        nrm: World space shading normal.
        wo: World space camera vector.
        wi: World space light vector
        alpha: Specular roughness parameter with shape [minibatch_size, height, width, 1]
        min_roughness: Scalar roughness clamping threshold

        use_python: Use PyTorch implementation (for validation)
    Returns:
        Shaded specular color
    '''

    if use_python:
        out = bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=min_roughness)
    else:
        out = _pbr_specular_func.apply(col, nrm, wo, wi, alpha, min_roughness)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_specular contains inf or NaN"
    return out

class _pbr_bsdf_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF):
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos)
        ctx.min_roughness = min_roughness
        ctx.BSDF = BSDF
        out = _get_plugin().pbr_bsdf_fwd(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        kd, arm, pos, nrm, view_pos, light_pos = ctx.saved_variables
        return _get_plugin().pbr_bsdf_bwd(kd, arm, pos, nrm, view_pos, light_pos, ctx.min_roughness, ctx.BSDF, dout) + (None, None, None)

def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08, bsdf="lambert", use_python=False):
    '''Physically-based bsdf, both diffuse & specular lobes
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        kd: Diffuse albedo.
        arm: Specular parameters (attenuation, linear roughness, metalness).
        pos: World space position.
        nrm: World space shading normal.
        view_pos: Camera position in world space, typically using broadcasting.
        light_pos: Light position in world space, typically using broadcasting.
        min_roughness: Scalar roughness clamping threshold
        bsdf: Controls diffuse BSDF, can be either 'lambert' or 'frostbite'

        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded color.
    '''    

    BSDF = 0 
    if bsdf == 'frostbite':
        BSDF = 1

    if use_python:
        out = bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)
    else:
        out = _pbr_bsdf_func.apply(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_bsdf contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# cubemap filter with filtering across edges

def _cube_to_dir(s, x, y):
    """Convert cubemap face coordinates to 3D direction vectors."""
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def _diffuse_cubemap_pytorch(cubemap):
    """Pure PyTorch implementation of diffuse irradiance cubemap convolution.

    Args:
        cubemap: [6, H, W, C] cubemap tensor
    Returns:
        [6, H, W, C] diffuse irradiance cubemap
    """
    import nvdiffrast.torch as dr

    faces, res, _, channels = cubemap.shape
    out = torch.zeros_like(cubemap)

    # Generate sample directions for integration (on the hemisphere)
    n_samples_theta = 64
    n_samples_phi = 64 * 2

    theta = torch.linspace(0, np.pi / 2.0, n_samples_theta, device=cubemap.device)
    phi = torch.linspace(0, 2.0 * np.pi, n_samples_phi, device=cubemap.device)
    d_theta = (np.pi / 2.0) / n_samples_theta
    d_phi = (2.0 * np.pi) / n_samples_phi

    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    # For each face and texel, compute the normal and integrate
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=cubemap.device),
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=cubemap.device),
            indexing='ij',
        )
        normal = torch.nn.functional.normalize(_cube_to_dir(s, gx, gy), dim=-1)  # [H, W, 3]

        # Build a tangent frame for each normal
        up = torch.zeros_like(normal)
        up[..., 1] = 1.0
        # Handle case where normal is parallel to up
        mask = (torch.abs(normal[..., 1]) > 0.999)
        up[mask, 0] = 1.0
        up[mask, 1] = 0.0

        tangent = torch.nn.functional.normalize(torch.cross(up, normal, dim=-1), dim=-1)
        bitangent = torch.cross(normal, tangent, dim=-1)

        irradiance = torch.zeros(res, res, channels, device=cubemap.device)

        for i in range(n_samples_theta):
            st = sintheta[i]
            ct = costheta[i]
            for j in range(n_samples_phi):
                sp = torch.sin(phi[j])
                cp = torch.cos(phi[j])

                # Hemisphere sample in tangent space -> world space
                sample_dir = (
                    tangent * (st * cp).unsqueeze(-1) +
                    bitangent * (st * sp).unsqueeze(-1) +
                    normal * ct
                )  # [H, W, 3]

                # Sample the cubemap
                color = dr.texture(
                    cubemap[None, ...],
                    sample_dir[None, ...].contiguous(),
                    filter_mode='linear',
                    boundary_mode='cube',
                )[0]  # [H, W, C]

                # cos(theta) * sin(theta) weighting
                irradiance = irradiance + color * (ct * st * d_theta * d_phi)

        out[s] = irradiance

    # Normalize: integral of cos(theta)*sin(theta) over hemisphere = pi
    # We already account for d_theta and d_phi in the sum
    return out

def _specular_cubemap_pytorch(cubemap, roughness, cutoff=0.99):
    """Pure PyTorch implementation of specular cubemap filtering using GGX importance sampling.

    Args:
        cubemap: [6, H, W, C] cubemap tensor
        roughness: float roughness value
        cutoff: energy cutoff for NDF bounds
    Returns:
        [6, H, W, C] filtered specular cubemap
    """
    import nvdiffrast.torch as dr

    faces, res, _, channels = cubemap.shape
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    out = torch.zeros(6, res, res, channels, device=cubemap.device, dtype=cubemap.dtype)

    n_samples = 512

    # Generate quasi-random samples using Hammersley sequence
    def radical_inverse(n):
        bits = 0
        factor = 0.5
        while n > 0:
            bits += (n % 2) * factor
            n //= 2
            factor *= 0.5
        return bits

    samples_xi = torch.tensor(
        [[float(i) / n_samples, radical_inverse(i)] for i in range(n_samples)],
        device=cubemap.device, dtype=cubemap.dtype
    )

    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=cubemap.device),
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=cubemap.device),
            indexing='ij',
        )
        normal = torch.nn.functional.normalize(_cube_to_dir(s, gx, gy), dim=-1)  # [H, W, 3]
        view = normal  # For pre-filtered environment maps, V = N = R

        # Build tangent frame
        up = torch.zeros_like(normal)
        up[..., 1] = 1.0
        mask = (torch.abs(normal[..., 1]) > 0.999)
        up[mask, 0] = 1.0
        up[mask, 1] = 0.0
        tangent = torch.nn.functional.normalize(torch.cross(up, normal, dim=-1), dim=-1)
        bitangent = torch.cross(normal, tangent, dim=-1)

        total_color = torch.zeros(res, res, channels, device=cubemap.device, dtype=cubemap.dtype)
        total_weight = torch.zeros(res, res, 1, device=cubemap.device, dtype=cubemap.dtype)

        for i in range(n_samples):
            xi1 = samples_xi[i, 0].item()
            xi2 = samples_xi[i, 1].item()

            # GGX importance sampling
            cos_theta = np.sqrt((1.0 - xi1) / (1.0 + (alpha2 - 1.0) * xi1))
            sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
            phi_val = 2.0 * np.pi * xi2

            # Half vector in tangent space -> world space
            h = (
                tangent * (sin_theta * np.cos(phi_val)) +
                bitangent * (sin_theta * np.sin(phi_val)) +
                normal * cos_theta
            )
            h = torch.nn.functional.normalize(h, dim=-1)

            # Reflect view around half vector to get light direction
            light = 2.0 * (view * h).sum(dim=-1, keepdim=True) * h - view
            light = torch.nn.functional.normalize(light, dim=-1)

            ndotl = (normal * light).sum(dim=-1, keepdim=True).clamp(min=0.0)

            # Only count samples where light is above the surface
            valid = (ndotl > 0.0).float()

            color = dr.texture(
                cubemap[None, ...],
                light[None, ...].contiguous(),
                filter_mode='linear',
                boundary_mode='cube',
            )[0]  # [H, W, C]

            total_color = total_color + color * ndotl * valid
            total_weight = total_weight + ndotl * valid

        # Normalize
        out[s] = torch.where(total_weight > 0, total_color / total_weight, torch.zeros_like(total_color))

    return out

class _diffuse_cubemap_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        out = _get_plugin().diffuse_cubemap_fwd(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, = ctx.saved_variables
        cubemap_grad = _get_plugin().diffuse_cubemap_bwd(cubemap, dout)
        return cubemap_grad, None

def diffuse_cubemap(cubemap, use_python=False):
    plugin = _get_plugin()
    if plugin is None or use_python:
        out = _diffuse_cubemap_pytorch(cubemap)
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
    if torch.is_anomaly_enabled():
        assert not torch.isnan(out).any(), "Output of diffuse_cubemap contains inf or NaN"
    return out

class _specular_cubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        out = _get_plugin().specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = _get_plugin().specular_cubemap_bwd(cubemap, bounds, dout, ctx.roughness, ctx.theta_cutoff)
        return cubemap_grad, None, None, None

# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    def ndfGGX(alphaSqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi/2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness**4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = _get_plugin().specular_bounds(res, costheta[idx])

    return costheta[idx], bounds
__ndfBoundsDict = {}

def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(cubemap.shape)

    plugin = _get_plugin()
    if plugin is None or use_python:
        out = _specular_cubemap_pytorch(cubemap, roughness, cutoff)
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
        out = out[..., 0:3] / out[..., 3:]
    if torch.is_anomaly_enabled():
        assert not torch.isnan(out).any(), "Output of specular_cubemap contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Fast image loss function

class _image_loss_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, target, loss, tonemapper):
        ctx.loss, ctx.tonemapper = loss, tonemapper
        ctx.save_for_backward(img, target)
        out = _get_plugin().image_loss_fwd(img, target, loss, tonemapper, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        img, target = ctx.saved_variables
        return _get_plugin().image_loss_bwd(img, target, dout, ctx.loss, ctx.tonemapper) + (None, None, None)

def image_loss(img, target, loss='l1', tonemapper='none', use_python=False):
    '''Compute HDR image loss. Combines tonemapping and loss into a single kernel for better perf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        img: Input image.
        target: Target (reference) image. 
        loss: Type of loss. Valid options are ['l1', 'mse', 'smape', 'relmse']
        tonemapper: Tonemapping operations. Valid options are ['none', 'log_srgb']
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Image space loss (scalar value).
    '''
    if use_python:
        out = image_loss_fn(img, target, loss, tonemapper)
    else:
        out = _image_loss_func.apply(img, target, loss, tonemapper)
        out = torch.sum(out) / (img.shape[0]*img.shape[1]*img.shape[2])

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of image_loss contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Transform points function

class _xfm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, isPoints):
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        return _get_plugin().xfm_fwd(points, matrix, isPoints, False)

    @staticmethod
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        return (_get_plugin().xfm_bwd(points, matrix, dout, ctx.isPoints),) + (None, None, None)

def xfm_points(points, matrix, use_python=False):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    
    if use_python:
        out = torch.matmul(torch.nn.functional.pad(points, pad=(0,1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    else:
        out = _xfm_func.apply(points, matrix, True)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def xfm_vectors(vectors, matrix, use_python=False):
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    

    if use_python:
        out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    else:
        out = _xfm_func.apply(vectors, matrix, False)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out



