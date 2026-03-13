/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include "ssr.h"
#include <cooperative_groups.h>
#include <math.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ============================================================
// Spherical Harmonics -> RGB  (unchanged)
// ============================================================
__device__ glm::vec3 computeColorFromSH(
	const int idx, const int deg, const int max_coeffs,
	const glm::vec3* means, glm::vec3 campos,
	const float* shs, bool* clamped)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x, y = dir.y, z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x*x, yy = y*y, zz = z*z;
			float xy = x*y, yz = y*z, xz = x*z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// ============================================================
// 2D / 3D covariance  (unchanged)
// ============================================================
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y,
	float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	float3 t = transformPoint4x3(mean, viewmatrix);
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z, tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	glm::mat3 T = W * J;
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(const glm::vec3 scale, float mod,
	const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;
	float r = q.x, x = q.y, y = q.z, z = q.w;
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y*y + z*z), 2.f * (x*y - r*z), 2.f * (x*z + r*y),
		2.f * (x*y + r*z), 1.f - 2.f * (x*x + z*z), 2.f * (y*z - r*x),
		2.f * (x*z - r*y), 2.f * (y*z + r*x), 1.f - 2.f * (x*x + y*y));
	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0]; cov3D[1] = Sigma[0][1]; cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1]; cov3D[4] = Sigma[1][2]; cov3D[5] = Sigma[2][2];
}

// ============================================================
// Preprocess kernel (unchanged)
// ============================================================
template<int C>
__global__ void preprocessCUDA(
	const int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	bool* clamped,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float3* pos_view,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	uint32_t* tiles_touched,
	const dim3 grid,
	const bool prefiltered,
	const bool cubemap)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	pos_view[idx] = p_view;

	float* cov3D;
	if (cov3D_precomp != nullptr) {
		cov3D = (float*)cov3D_precomp + 6 * idx;
	} else {
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + 6 * idx);
		cov3D = cov3Ds + 6 * idx;
	}

	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy,
	                           cov3D, viewmatrix);

	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(lambda1));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, grid, rect_min, rect_max);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M,
			(glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// ============================================================
// Lite render kernel (unchanged)
// ============================================================
template<uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
liteRenderCUDA(
	const int W, int H,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* __restrict__ features,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ depth,
	const float* __restrict__ bg_color,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ final_T,
	float* __restrict__ out_color,
	float* __restrict__ out_opacity,
	float* __restrict__ out_depth,
	bool argmax_depth)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0, last_contributor = 0;
	float C[CHANNELS] = { 0.0f };
	float D = 0.0f, O = 0.0f, max_weight = 0.0f, except_depth = 0.0f;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE) break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) { done = true; continue; }
			const float weight = alpha * T;
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * weight;
			D += depth[collected_id[j]] * weight;
			O += weight;
			if (weight > max_weight) { except_depth = depth[collected_id[j]]; max_weight = weight; }
			T = test_T;
			last_contributor = contributor;
		}
	}

	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = (O > 1e-6f) ? (argmax_depth ? except_depth : D / O) : 0.0f;
		out_opacity[pix_id] = O;
	}
}

// ============================================================
// OPTIMIZED renderCUDA: writes per-bucket T/color snapshots
// and records max_contrib per tile for the backward pass.
// ============================================================
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const int W, int H,
	const float fx, float fy,
	const float* means3D,
	const float* cam_pos,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* viewmatrix,
	const float* __restrict__ features,
	const float* __restrict__ normals,
	const float* __restrict__ albedo,
	const float* __restrict__ roughness,
	const float* __restrict__ metallic,
	const float3* __restrict__ pos_view,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ depth,
	const float* __restrict__ bg_color,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ final_T,
	float* __restrict__ out_color,
	float* __restrict__ out_opacity,
	float* __restrict__ out_depth,
	float* __restrict__ out_normal,
	float* __restrict__ out_normal_view,
	float* __restrict__ out_pos,
	float* __restrict__ out_albedo,
	float* __restrict__ out_roughness,
	float* __restrict__ out_metallic,
	// --- SampleState ---
	uint32_t* __restrict__ bucket_to_tile,
	float*    __restrict__ sampled_T,
	float*    __restrict__ sampled_ar,
	uint32_t* __restrict__ max_contrib_tile,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	// -------------------
	bool argmax_depth,
	bool inference)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	const float cx = float(W) / 2.0f, cy = float(H) / 2.0f;
	const float3 ray = { (pixf.x - cx) / fx, (pixf.y - cy) / fy, 1.0f };

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	// Tile and bucket base for snapshot writing
	const uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	const uint32_t bucket_base = (tile_id == 0) ? 0u : per_tile_bucket_offset[tile_id - 1];

	uint2 range = ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Per-pixel rendering state
	float T = 1.0f;
	uint32_t contributor = 0, last_contributor = 0;
	float C[CHANNELS] = { 0.0f };
	float N[CHANNELS] = { 0.0f };
	float A[CHANNELS] = { 0.0f };
	float R = 0.0f, M_val = 0.0f, D = 0.0f;
	float3 POS = { 0.0f, 0.0f, 0.0f };
	float3 N_world = { 0.0f, 0.0f, 0.0f };
	float3 N_view  = { 0.0f, 0.0f, 0.0f };
	float O = 0.0f, max_weight = 0.0f, except_depth = 0.0f;
	float3 except_pos = { 0.0f, 0.0f, 0.0f };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE) break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			// ---- Snapshot at bucket boundary (every 32 splats) ----
			const int splat_in_tile = i * BLOCK_SIZE + j;
			if (splat_in_tile % 32 == 0) {
				const uint32_t bucket_in_tile  = (uint32_t)(splat_in_tile / 32);
				const uint32_t global_bucket   = bucket_base + bucket_in_tile;
				// Lane 0 records tile association
				if (block.thread_rank() == 0)
					bucket_to_tile[global_bucket] = tile_id;
				// All threads write their transmittance and accumulated colour
				sampled_T[global_bucket * BLOCK_SIZE + block.thread_rank()] = T;
				for (uint32_t ch = 0; ch < CHANNELS; ch++) {
					sampled_ar[global_bucket * BLOCK_SIZE * CHANNELS
					           + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];
				}
			}

			if (done) continue;   // done threads skip blend but helped with snapshot

			contributor++;

			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;

			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) continue;
			float test_T = T * (1.0f - alpha);
			if (test_T < 0.0001f) { done = true; continue; }

			float3 view_dir = {
				cam_pos[0] - means3D[collected_id[j] * 3 + 0],
				cam_pos[1] - means3D[collected_id[j] * 3 + 1],
				cam_pos[2] - means3D[collected_id[j] * 3 + 2],
			};

			const float weight = alpha * T;
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * weight;
				A[ch] += albedo[collected_id[j] * CHANNELS + ch] * weight;
				N[ch] += normals[collected_id[j] * CHANNELS + ch] * weight;
			}
			R     += roughness[collected_id[j]] * weight;
			M_val += metallic[collected_id[j]] * weight;
			D     += depth[collected_id[j]] * weight;
			POS.x += pos_view[collected_id[j]].x * weight;
			POS.y += pos_view[collected_id[j]].y * weight;
			POS.z += pos_view[collected_id[j]].z * weight;
			O     += weight;

			if (weight > max_weight) {
				except_depth = depth[collected_id[j]];
				except_pos   = { pos_view[collected_id[j]].x,
				                 pos_view[collected_id[j]].y,
				                 pos_view[collected_id[j]].z };
				max_weight = weight;
			}

			T = test_T;
			last_contributor = contributor;
		}
	}

	// ---- Write outputs ----
	if (inside)
	{
		final_T[pix_id]   = T;
		n_contrib[pix_id] = last_contributor;

		N_world = { N[0], N[1], N[2] };
		N_view  = transformVec4x3(N_world, viewmatrix);
		N_view  = normalize(N_view);
		out_normal_view[pix_id]             = N_view.x;
		out_normal_view[1 * H * W + pix_id] = N_view.y;
		out_normal_view[2 * H * W + pix_id] = N_view.z;

		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id]  = C[ch] + T * bg_color[ch];
			out_normal[ch * H * W + pix_id] = N[ch];
			out_albedo[ch * H * W + pix_id] = A[ch];
		}
		out_roughness[pix_id] = inference ? (R + T) : R;
		out_metallic[pix_id]  = M_val;

		if (O > 1e-6f) {
			out_depth[pix_id]              = argmax_depth ? except_depth : D / O;
			out_pos[pix_id]                = argmax_depth ? except_pos.x : POS.x / O;
			out_pos[1 * H * W + pix_id]   = argmax_depth ? except_pos.y : POS.y / O;
			out_pos[2 * H * W + pix_id]   = argmax_depth ? except_pos.z : POS.z / O;
		} else {
			out_depth[pix_id]             = 0.0f;
			out_pos[pix_id]               = 0.0f;
			out_pos[1 * H * W + pix_id]  = 0.0f;
			out_pos[2 * H * W + pix_id]  = 0.0f;
		}
		out_opacity[pix_id] = O;
	}

	// ---- Tile-level max_contrib (for backward early stopping) ----
	__shared__ uint32_t smem_max;
	if (block.thread_rank() == 0) smem_max = 0u;
	block.sync();
	if (inside)
		atomicMax(&smem_max, last_contributor);
	block.sync();
	if (block.thread_rank() == 0)
		max_contrib_tile[tile_id] = smem_max;
}

// ============================================================
// renderFeatureCUDA  (unchanged from original)
// ============================================================
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderFeatureCUDA(
	const int W, int H,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* __restrict__ features,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const int feature_dim,
	float* __restrict__ out_feature)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	float T = 1.0f;
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE) break;
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) {
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) { done = true; continue; }
			const float weight = alpha * T;
			const int point_offset = collected_id[j] * feature_dim;
			if (inside)
				for (int feat_ch = 0; feat_ch < feature_dim; ++feat_ch)
					out_feature[feat_ch * H * W + pix_id] += features[point_offset + feat_ch] * weight;
			T = test_T;
		}
	}
}

// ============================================================
// SSAO / SSR / depthToNormal kernels  (unchanged)
// ============================================================
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
SSAOCUDA(
	int W, int H,
	const float focal_x, const float focal_y,
	const float radius, const float bias,
	const float thick, const float delta,
	const int step, const int start,
	const float* __restrict__ out_normal,
	const float* __restrict__ out_pos,
	float* __restrict__ occlusion)
{
	auto block = cg::this_thread_block();
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	if (pix.x > W-1 || pix.y > H-1) return;

	float3 normal_un = { out_normal[pix_id], out_normal[1*H*W+pix_id], out_normal[2*H*W+pix_id] };
	float3 normal = normalize(normal_un);
	float3 pos    = { out_pos[pix_id], out_pos[1*H*W+pix_id], out_pos[2*H*W+pix_id] };
	float3 up     = { 0.0f, 1.0f, 0.0f };
	float rndot   = dot(up, normal);
	float3 untangent = { up.x - normal.x*rndot, up.y - normal.y*rndot, up.z - normal.z*rndot };
	float3 tangent   = normalize(untangent);
	float3 bitangent = normalize(cross(normal, tangent));
	float TBN[9] = { tangent.x, tangent.y, tangent.z,
	                 bitangent.x, bitangent.y, bitangent.z,
	                 normal.x, normal.y, normal.z };
	float occ = 0.0f, nrSamples = 0.0f;
	float sampleDelta = delta * M_PIf;
	for (float phi = 0.0f; phi < 2.0f * M_PIf; phi += sampleDelta) {
		for (float theta = 0.0f; theta <= 0.5f * M_PIf; theta += sampleDelta * 0.5f) {
			float cosh = cosf(theta);
			float3 ts = { sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta) };
			ts = normalize(ts);
			float3 sv = transformVec3x3(ts, TBN);
			float3 sp = { 0.0f, 0.0f, 0.0f };
			nrSamples += cosh * sinf(theta);
			for (int j = start; j < step; ++j) {
				sp.x = pos.x + sv.x*j*(1+pos.z/100)*(1+pos.z/100)*radius/step;
				sp.y = pos.y + sv.y*j*(1+pos.z/100)*(1+pos.z/100)*radius/step;
				sp.z = pos.z + sv.z*j*(1+pos.z/100)*(1+pos.z/100)*radius/step;
				float cx2 = float(W)/2.0f, cy2 = float(H)/2.0f;
				int2 depth_id = get_coord(cx2, cy2, focal_x, focal_y, sp);
				if (depth_id.x < 0 || depth_id.x > W-1 || depth_id.y < 0 || depth_id.y > H-1) break;
				float sd = out_pos[2*H*W + W*depth_id.y + depth_id.x];
				if (sd <= sp.z + bias && sd >= sp.z - thick) { occ += cosh*sinf(theta); break; }
			}
		}
	}
	occlusion[pix_id] = (nrSamples > 0.0f)
		? fmaxf(0.0f, fminf(1.0f, 1.0f - (occ / nrSamples)))
		: 1.0f;
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
SSRCUDA(
	int W, int H,
	const float focal_x, const float focal_y,
	const float radius, const float bias,
	const float thick, const float delta,
	const int step, const int start,
	const float* __restrict__ out_normal,
	const float* __restrict__ out_pos,
	const float* __restrict__ out_rgb,
    const float* __restrict__ out_albedo,
    const float* __restrict__ out_roughness,
    const float* __restrict__ out_metallic,
    const float* __restrict__ out_F0,
	float* __restrict__ color,
	float* __restrict__ abd)
{
	auto block = cg::this_thread_block();
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	if (pix.x > W-1 || pix.y > H-1) return;

	float3 pos = { out_pos[pix_id], out_pos[1*H*W+pix_id], out_pos[2*H*W+pix_id] };
	float3 gd  = { 0.0f, 0.0f, 0.0f };
	float3 diffuse = { 0.0f, 0.0f, 0.0f };
	float3 normal_un = { out_normal[pix_id], out_normal[1*H*W+pix_id], out_normal[2*H*W+pix_id] };
	float3 normal    = normalize(normal_un);
	float3 N = normal;
	float3 up = { 0.0f, 1.0f, 0.0f };
	float rndot = dot(up, normal);
	float3 untangent = { up.x - normal.x*rndot, up.y - normal.y*rndot, up.z - normal.z*rndot };
	float3 tangent   = normalize(untangent);
	float3 bitangent = normalize(cross(normal, tangent));
	float TBN[9] = { tangent.x, tangent.y, tangent.z,
	                 bitangent.x, bitangent.y, bitangent.z,
	                 normal.x, normal.y, normal.z };

	float3 albedo3 = { out_albedo[pix_id], out_albedo[1*H*W+pix_id], out_albedo[2*H*W+pix_id] };
	float3 F0_3    = { out_F0[pix_id], out_F0[1*H*W+pix_id], out_F0[2*H*W+pix_id] };
	float roughness_val = out_roughness[pix_id];
	float metallic_val  = out_metallic[pix_id];
	float3 V = normalize(-pos);
	float3 F = fresnelSchlick(fmaxf(dot(N, V), 0.0000001f), F0_3);
	float3 kS = F;
	float3 kD = { 1.0f - kS.x, 1.0f - kS.y, 1.0f - kS.z };
	kD.x *= 1.0f - metallic_val;
	kD.y *= 1.0f - metallic_val;
	kD.z *= 1.0f - metallic_val;

	float sampleDelta = delta * M_PIf;
	float nrSamples = 0.0f;
	for (float phi = 0.0f; phi < 2.0f*M_PIf; phi += sampleDelta) {
		for (float theta = 0.0f; theta <= 0.5f*M_PIf; theta += sampleDelta*0.5f) {
			float3 ts = { sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta) };
			ts = normalize(ts);
			float3 sv = transformVec3x3(ts, TBN);
			float3 sp = { 0.0f, 0.0f, 0.0f };
			nrSamples += 1.0f;
			for (int j = start; j < step; ++j) {
				sp.x = pos.x + sv.x*j*(1+pos.z/100)*(1+pos.z/100)*radius/step;
				sp.y = pos.y + sv.y*j*(1+pos.z/100)*(1+pos.z/100)*radius/step;
				sp.z = pos.z + sv.z*j*(1+pos.z/100)*(1+pos.z/100)*radius/step;
				float cx2 = float(W)/2.0f, cy2 = float(H)/2.0f;
				int2 depth_id = get_coord(cx2, cy2, focal_x, focal_y, sp);
				if (depth_id.x < 0 || depth_id.x > W-1 || depth_id.y < 0 || depth_id.y > H-1) break;
				float3 rgb3 = { out_rgb[W*depth_id.y+depth_id.x],
				                out_rgb[H*W+W*depth_id.y+depth_id.x],
				                out_rgb[2*H*W+W*depth_id.y+depth_id.x] };
				float sd = out_pos[2*H*W + W*depth_id.y + depth_id.x];
				if (sd <= sp.z + bias && sd >= sp.z - thick) {
					diffuse.x += rgb3.x * cosf(theta) * sinf(theta);
					diffuse.y += rgb3.y * cosf(theta) * sinf(theta);
					diffuse.z += rgb3.z * cosf(theta) * sinf(theta);
					break;
				}
			}
		}
	}
	if (nrSamples > 0.0f) {
		gd.x = M_PIf * diffuse.x * (1.0f/nrSamples) * kD.x;
		gd.y = M_PIf * diffuse.y * (1.0f/nrSamples) * kD.y;
		gd.z = M_PIf * diffuse.z * (1.0f/nrSamples) * kD.z;
		diffuse.x = gd.x * albedo3.x;
		diffuse.y = gd.y * albedo3.y;
		diffuse.z = gd.z * albedo3.z;
	} else {
		diffuse = { 1e-7f, 1e-7f, 1e-7f };
		gd      = { 1e-7f, 1e-7f, 1e-7f };
	}
	color[pix_id]           = diffuse.x;
	color[1*H*W + pix_id]   = diffuse.y;
	color[2*H*W + pix_id]   = diffuse.z;
	abd[pix_id]             = gd.x;
	abd[1*H*W + pix_id]     = gd.y;
	abd[2*H*W + pix_id]     = gd.z;
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
depthmapToNormalCUDA(
	int W, int H,
	const float focal_x, const float focal_y,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ out_depth,
	float* __restrict__ normal_from_depth,
	float* __restrict__ depth_pos)
{
	auto block = cg::this_thread_block();
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;

	if (pix.x <= 0 || pix.x >= W-1 || pix.y <= 0 || pix.y >= H-1) return;

	const float depth_thresh = 0.01f;
	const float depth = out_depth[pix_id];
	float cx = float(W)/2.0f, cy = float(H)/2.0f;
	float3 pos = get_position(pix.x, pix.y, cx, cy, focal_x, focal_y, depth);
	depth_pos[pix_id]           = pos.x;
	depth_pos[1*H*W + pix_id]  = pos.y;
	depth_pos[2*H*W + pix_id]  = pos.z;
	if (depth < depth_thresh) return;

	int pad = 2;
	for (int x = -pad; x < pad+1; ++x) {
		if (int(pix.x+x) < 0 || int(pix.x+x) > W-1) return;
		for (int y = -pad; y < pad+1; ++y) {
			if (int(pix.y+y) < 0 || int(pix.y+y) > H-1) return;
			if (out_depth[pix_id + W*y + x] < depth_thresh) return;
		}
	}
	float da = out_depth[pix_id-W], db = out_depth[pix_id+1];
	float dc = out_depth[pix_id+W], dd = out_depth[pix_id-1];
	float dab = out_depth[pix_id-W+1], dbc = out_depth[pix_id+W+1];
	float dcd = out_depth[pix_id+W-1], dda = out_depth[pix_id-W-1];
	float3 pa = get_position(pix.x, pix.y-1, cx, cy, focal_x, focal_y, da);
	float3 pb = get_position(pix.x+1, pix.y, cx, cy, focal_x, focal_y, db);
	float3 pc = get_position(pix.x, pix.y+1, cx, cy, focal_x, focal_y, dc);
	float3 pd = get_position(pix.x-1, pix.y, cx, cy, focal_x, focal_y, dd);
	float3 pab = get_position(pix.x+1, pix.y-1, cx, cy, focal_x, focal_y, dab);
	float3 pbc = get_position(pix.x+1, pix.y+1, cx, cy, focal_x, focal_y, dbc);
	float3 pcd = get_position(pix.x-1, pix.y+1, cx, cy, focal_x, focal_y, dcd);
	float3 pda = get_position(pix.x-1, pix.y-1, cx, cy, focal_x, focal_y, dda);
	float3 ea = pda - pab, eb = pab - pbc, ec = pbc - pcd, ed = pcd - pda;
	float3 eac = pc - pa, ebd = pd - pb;
	float3 ecdab = pab - pcd, ebcad = pda - pbc;
	float3 n1 = cross(ea, ed), n2 = cross(ed, ec);
	float3 n3 = cross(ec, eb), n4 = cross(eb, ea);
	float3 n5 = cross(eac, ebd), n6 = cross(ebcad, ecdab);
	float3 n = (normalize(n1)+normalize(n2)+normalize(n3)+normalize(n4)+normalize(n5)+normalize(n6))/6.0f;
	float nx = viewmatrix[0]*n.x + viewmatrix[1]*n.y + viewmatrix[2]*n.z;
	float ny = viewmatrix[4]*n.x + viewmatrix[5]*n.y + viewmatrix[6]*n.z;
	float nz = viewmatrix[8]*n.x + viewmatrix[9]*n.y + viewmatrix[10]*n.z;
	normal_from_depth[pix_id]          = nx;
	normal_from_depth[1*H*W + pix_id]  = ny;
	normal_from_depth[2*H*W + pix_id]  = nz;
}

// ============================================================
// FORWARD namespace wrapper functions
// ============================================================

void FORWARD::render_feature(
	const dim3 grid, dim3 block,
	const int W, int H,
	const uint2* ranges, const uint32_t* point_list,
	const float* features, const float2* means2D,
	const float4* conic_opacity, const int feature_dim, float* out_feature)
{
	renderFeatureCUDA<<<grid, block>>>(W, H, ranges, point_list, features,
		means2D, conic_opacity, feature_dim, out_feature);
}

void FORWARD::lite_render(
	const dim3 grid, dim3 block, int W, int H,
	const uint2* ranges, const uint32_t* point_list,
	const float* colors, const float2* means2D,
	const float4* conic_opacity, const float* depth,
	const float* bg_color, uint32_t* n_contrib, float* final_T,
	float* out_color, float* out_opacity, float* out_depth, bool argmax_depth)
{
	liteRenderCUDA<NUM_CHANNELS><<<grid, block>>>(W, H, ranges, point_list,
		colors, means2D, conic_opacity, depth, bg_color,
		n_contrib, final_T, out_color, out_opacity, out_depth, argmax_depth);
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const int W, int H, const float fx, float fy,
	const float* means3D, const float* cam_pos,
	const uint2* ranges, const uint32_t* point_list,
	const float* viewmatrix, const float* features,
	const float* normal, const float* albedo,
	const float* roughness, const float* metallic,
	const float3* pos_view, const float2* means2D,
	const float4* conic_opacity, const float* depth,
	const float* bg_color, uint32_t* n_contrib, float* final_T,
	float* out_color, float* out_opacity, float* out_depth,
	float* out_normal, float* out_normal_view, float* out_pos,
	float* out_albedo, float* out_roughness, float* out_metallic,
	// --- SampleState ---
	uint32_t* bucket_to_tile, float* sampled_T, float* sampled_ar,
	uint32_t* max_contrib_tile, const uint32_t* per_tile_bucket_offset,
	// -------------------
	const bool argmax_depth, const bool inference)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		W, H, fx, fy, means3D, cam_pos,
		ranges, point_list, viewmatrix,
		features, normal, albedo, roughness, metallic,
		pos_view, means2D, conic_opacity, depth, bg_color,
		n_contrib, final_T,
		out_color, out_opacity, out_depth,
		out_normal, out_normal_view, out_pos,
		out_albedo, out_roughness, out_metallic,
		bucket_to_tile, sampled_T, sampled_ar,
		max_contrib_tile, per_tile_bucket_offset,
		argmax_depth, inference);
}

void FORWARD::preprocess(
	const int P, int D, int M,
	const float* means3D, const glm::vec3* scales,
	const float scale_modifier, const glm::vec4* rotations,
	const float* opacities, const float* shs,
	const float* cov3D_precomp, const float* colors_precomp,
	const float* viewmatrix, const float* projmatrix,
	const glm::vec3* cam_pos, const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii, bool* clamped, float2* means2D,
	float* depths, float3* pos_view, float* cov3Ds,
	float* rgb, float4* conic_opacity,
	uint32_t* tiles_touched, const dim3 grid,
	const bool prefiltered, const bool cubemap)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>> (
		P, D, M, means3D, scales, scale_modifier, rotations,
		opacities, shs, cov3D_precomp, colors_precomp,
		viewmatrix, projmatrix, cam_pos,
		W, H, tan_fovx, tan_fovy, focal_x, focal_y,
		clamped, radii, means2D, depths, pos_view,
		cov3Ds, rgb, conic_opacity, tiles_touched,
		grid, prefiltered, cubemap);
}

void FORWARD::depthToNormal(
	const dim3 grid, const dim3 block,
	const int W, int H, const float focal_x, const float focal_y,
	const float* viewmatrix, const float* depthMap,
	float* normalMap, float* depth_pos)
{
	depthmapToNormalCUDA<<<grid, block>>>(
		W, H, focal_x, focal_y, viewmatrix, depthMap, normalMap, depth_pos);
}

void FORWARD::SSAO(
	const dim3 grid, const dim3 block, int W, int H,
	const float focal_x, const float focal_y,
	const float radius, const float bias, const float thick,
	const float delta, const int step, const int start,
	const float* out_normal, const float* out_pos, float* occlusion)
{
	SSAOCUDA<<<grid, block>>>(W, H, focal_x, focal_y,
		radius, bias, thick, delta, step, start,
		out_normal, out_pos, occlusion);
}

void FORWARD::SSR(
	const dim3 grid, const dim3 block, int W, int H,
	const float focal_x, const float focal_y,
	const float radius, const float bias, const float thick,
	const float delta, const int step, const int start,
	const float* out_normal, const float* out_pos,
	const float* out_rgb, const float* out_albedo,
    const float* out_roughness, const float* out_metallic,
    const float* out_F0, float* color, float* abd)
{
	SSRCUDA<<<grid, block>>>(W, H, focal_x, focal_y,
		radius, bias, thick, delta, step, start,
		out_normal, out_pos, out_rgb, out_albedo,
		out_roughness, out_metallic, out_F0, color, abd);
}
