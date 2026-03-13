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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "ssr.h"
namespace cg = cooperative_groups;

// ============================================================
// SH backward  (unchanged)
// ============================================================
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs,
	const glm::vec3* means, glm::vec3 campos, const float* shs,
	const bool* clamped, const glm::vec3* dL_dcolor,
	glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3*idx+0] ? 0 : 1;
	dL_dRGB.y *= clamped[3*idx+1] ? 0 : 1;
	dL_dRGB.z *= clamped[3*idx+2] ? 0 : 1;

	glm::vec3 dRGBdx(0,0,0), dRGBdy(0,0,0), dRGBdz(0,0,0);
	float x = dir.x, y = dir.y, z = dir.z;
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0) {
		dL_dsh[1] = -SH_C1 * y * dL_dRGB;
		dL_dsh[2] =  SH_C1 * z * dL_dRGB;
		dL_dsh[3] = -SH_C1 * x * dL_dRGB;
		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz =  SH_C1 * sh[2];
		if (deg > 1) {
			float xx=x*x, yy=y*y, zz=z*z, xy=x*y, yz=y*z, xz=x*z;
			dL_dsh[4] = SH_C2[0]*xy*dL_dRGB;
			dL_dsh[5] = SH_C2[1]*yz*dL_dRGB;
			dL_dsh[6] = SH_C2[2]*(2.f*zz-xx-yy)*dL_dRGB;
			dL_dsh[7] = SH_C2[3]*xz*dL_dRGB;
			dL_dsh[8] = SH_C2[4]*(xx-yy)*dL_dRGB;
			dRGBdx += SH_C2[0]*y*sh[4] + SH_C2[2]*2.f*-x*sh[6] + SH_C2[3]*z*sh[7] + SH_C2[4]*2.f*x*sh[8];
			dRGBdy += SH_C2[0]*x*sh[4] + SH_C2[1]*z*sh[5] + SH_C2[2]*2.f*-y*sh[6] + SH_C2[4]*2.f*-y*sh[8];
			dRGBdz += SH_C2[1]*y*sh[5] + SH_C2[2]*2.f*2.f*z*sh[6] + SH_C2[3]*x*sh[7];
			if (deg > 2) {
				dL_dsh[9]  = SH_C3[0]*y*(3.f*xx-yy)*dL_dRGB;
				dL_dsh[10] = SH_C3[1]*xy*z*dL_dRGB;
				dL_dsh[11] = SH_C3[2]*y*(4.f*zz-xx-yy)*dL_dRGB;
				dL_dsh[12] = SH_C3[3]*z*(2.f*zz-3.f*xx-3.f*yy)*dL_dRGB;
				dL_dsh[13] = SH_C3[4]*x*(4.f*zz-xx-yy)*dL_dRGB;
				dL_dsh[14] = SH_C3[5]*z*(xx-yy)*dL_dRGB;
				dL_dsh[15] = SH_C3[6]*x*(xx-3.f*yy)*dL_dRGB;
				dRGBdx += (SH_C3[0]*sh[9]*3.f*2.f*xy + SH_C3[1]*sh[10]*yz + SH_C3[2]*sh[11]*-2.f*xy + SH_C3[3]*sh[12]*-3.f*2.f*xz + SH_C3[4]*sh[13]*(-3.f*xx+4.f*zz-yy) + SH_C3[5]*sh[14]*2.f*xz + SH_C3[6]*sh[15]*3.f*(xx-yy));
				dRGBdy += (SH_C3[0]*sh[9]*3.f*(xx-yy) + SH_C3[1]*sh[10]*xz + SH_C3[2]*sh[11]*(-3.f*yy+4.f*zz-xx) + SH_C3[3]*sh[12]*-3.f*2.f*yz + SH_C3[4]*sh[13]*-2.f*xy + SH_C3[5]*sh[14]*-2.f*yz + SH_C3[6]*sh[15]*-3.f*2.f*xy);
				dRGBdz += (SH_C3[1]*sh[10]*xy + SH_C3[2]*sh[11]*4.f*2.f*yz + SH_C3[3]*sh[12]*3.f*(2.f*zz-xx-yy) + SH_C3[4]*sh[13]*4.f*2.f*xz + SH_C3[5]*sh[14]*(xx-yy));
			}
		}
	}
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
	float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z},
	                            float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// ============================================================
// 2D covariance backward  (unchanged)
// ============================================================
__global__ void computeCov2DCUDA(int P,
	const float3* means, const int* radii, const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics, const float* dL_depth,
	float3* dL_dmeans, float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0)) return;

	const float* cov3D = cov3Ds + 6 * idx;
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4*idx], dL_dconics[4*idx+1], dL_dconics[4*idx+3] };
	float3 t = transformPoint4x3(mean, view_matrix);

	const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z, tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x/t.z, 0, -(h_x*t.x)/(t.z*t.z),
	                         0, h_y/t.z, -(h_y*t.y)/(t.z*t.z), 0, 0, 0);
	glm::mat3 W = glm::mat3(view_matrix[0],view_matrix[4],view_matrix[8],
	                         view_matrix[1],view_matrix[5],view_matrix[9],
	                         view_matrix[2],view_matrix[6],view_matrix[10]);
	glm::mat3 Vrk = glm::mat3(cov3D[0],cov3D[1],cov3D[2],
	                           cov3D[1],cov3D[3],cov3D[4],
	                           cov3D[2],cov3D[4],cov3D[5]);
	glm::mat3 T = W * J;
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;
	float a = cov2D[0][0] += 0.3f, b = cov2D[0][1], c = cov2D[1][1] += 0.3f;
	float denom = a*c - b*b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom*denom) + 0.0000001f);
	if (denom2inv != 0) {
		dL_da = denom2inv*(-c*c*dL_dconic.x + 2*b*c*dL_dconic.y + (denom-a*c)*dL_dconic.z);
		dL_dc = denom2inv*(-a*a*dL_dconic.z + 2*a*b*dL_dconic.y + (denom-a*c)*dL_dconic.x);
		dL_db = denom2inv*2*(b*c*dL_dconic.x - (denom+2*b*b)*dL_dconic.y + a*b*dL_dconic.z);
		dL_dcov[6*idx+0] = T[0][0]*T[0][0]*dL_da + T[0][0]*T[1][0]*dL_db + T[1][0]*T[1][0]*dL_dc;
		dL_dcov[6*idx+3] = T[0][1]*T[0][1]*dL_da + T[0][1]*T[1][1]*dL_db + T[1][1]*T[1][1]*dL_dc;
		dL_dcov[6*idx+5] = T[0][2]*T[0][2]*dL_da + T[0][2]*T[1][2]*dL_db + T[1][2]*T[1][2]*dL_dc;
		dL_dcov[6*idx+1] = 2*T[0][0]*T[0][1]*dL_da + (T[0][0]*T[1][1]+T[0][1]*T[1][0])*dL_db + 2*T[1][0]*T[1][1]*dL_dc;
		dL_dcov[6*idx+2] = 2*T[0][0]*T[0][2]*dL_da + (T[0][0]*T[1][2]+T[0][2]*T[1][0])*dL_db + 2*T[1][0]*T[1][2]*dL_dc;
		dL_dcov[6*idx+4] = 2*T[0][2]*T[0][1]*dL_da + (T[0][1]*T[1][2]+T[0][2]*T[1][1])*dL_db + 2*T[1][1]*T[1][2]*dL_dc;
	} else {
		for (int i = 0; i < 6; i++) dL_dcov[6*idx+i] = 0;
	}

	float dL_dT00 = 2*(T[0][0]*Vrk[0][0]+T[0][1]*Vrk[0][1]+T[0][2]*Vrk[0][2])*dL_da + (T[1][0]*Vrk[0][0]+T[1][1]*Vrk[0][1]+T[1][2]*Vrk[0][2])*dL_db;
	float dL_dT01 = 2*(T[0][0]*Vrk[1][0]+T[0][1]*Vrk[1][1]+T[0][2]*Vrk[1][2])*dL_da + (T[1][0]*Vrk[1][0]+T[1][1]*Vrk[1][1]+T[1][2]*Vrk[1][2])*dL_db;
	float dL_dT02 = 2*(T[0][0]*Vrk[2][0]+T[0][1]*Vrk[2][1]+T[0][2]*Vrk[2][2])*dL_da + (T[1][0]*Vrk[2][0]+T[1][1]*Vrk[2][1]+T[1][2]*Vrk[2][2])*dL_db;
	float dL_dT10 = 2*(T[1][0]*Vrk[0][0]+T[1][1]*Vrk[0][1]+T[1][2]*Vrk[0][2])*dL_dc + (T[0][0]*Vrk[0][0]+T[0][1]*Vrk[0][1]+T[0][2]*Vrk[0][2])*dL_db;
	float dL_dT11 = 2*(T[1][0]*Vrk[1][0]+T[1][1]*Vrk[1][1]+T[1][2]*Vrk[1][2])*dL_dc + (T[0][0]*Vrk[1][0]+T[0][1]*Vrk[1][1]+T[0][2]*Vrk[1][2])*dL_db;
	float dL_dT12 = 2*(T[1][0]*Vrk[2][0]+T[1][1]*Vrk[2][1]+T[1][2]*Vrk[2][2])*dL_dc + (T[0][0]*Vrk[2][0]+T[0][1]*Vrk[2][1]+T[0][2]*Vrk[2][2])*dL_db;
	float dL_dJ00 = W[0][0]*dL_dT00 + W[0][1]*dL_dT01 + W[0][2]*dL_dT02;
	float dL_dJ02 = W[2][0]*dL_dT00 + W[2][1]*dL_dT01 + W[2][2]*dL_dT02;
	float dL_dJ11 = W[1][0]*dL_dT10 + W[1][1]*dL_dT11 + W[1][2]*dL_dT12;
	float dL_dJ12 = W[2][0]*dL_dT10 + W[2][1]*dL_dT11 + W[2][2]*dL_dT12;
	float tz = 1.f/t.z, tz2 = tz*tz, tz3 = tz2*tz;
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x*tz2*dL_dJ00 - h_y*tz2*dL_dJ11 + (2*h_x*t.x)*tz3*dL_dJ02 + (2*h_y*t.y)*tz3*dL_dJ12;
	float3 dL_dmean = transformVec4x3Transpose({dL_dtx, dL_dty, dL_dtz}, view_matrix);
	dL_dmean.x += view_matrix[2]*dL_depth[idx];
	dL_dmean.y += view_matrix[6]*dL_depth[idx];
	dL_dmean.z += view_matrix[10]*dL_depth[idx];
	dL_dmeans[idx] = dL_dmean;
}

// ============================================================
// 3D covariance backward  (unchanged)
// ============================================================
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod,
	const glm::vec4 rot, const float* dL_dcov3Ds,
	glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	glm::vec4 q = rot;
	float r=q.x, x=q.y, y=q.z, z=q.w;
	glm::mat3 R = glm::mat3(
		1.f-2.f*(y*y+z*z), 2.f*(x*y-r*z), 2.f*(x*z+r*y),
		2.f*(x*y+r*z), 1.f-2.f*(x*x+z*z), 2.f*(y*z-r*x),
		2.f*(x*z-r*y), 2.f*(y*z+r*x), 1.f-2.f*(x*x+y*y));
	glm::mat3 S = glm::mat3(1.0f);
	glm::vec3 s = mod * scale;
	S[0][0]=s.x; S[1][1]=s.y; S[2][2]=s.z;
	glm::mat3 M = S * R;
	const float* dL_dcov3D = dL_dcov3Ds + 6*idx;
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f*dL_dcov3D[1], 0.5f*dL_dcov3D[2],
		0.5f*dL_dcov3D[1], dL_dcov3D[3], 0.5f*dL_dcov3D[4],
		0.5f*dL_dcov3D[2], 0.5f*dL_dcov3D[4], dL_dcov3D[5]);
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);
	dL_dMt[0] *= s.x; dL_dMt[1] *= s.y; dL_dMt[2] *= s.z;
	glm::vec4 dL_dq;
	dL_dq.x = 2*z*(dL_dMt[0][1]-dL_dMt[1][0]) + 2*y*(dL_dMt[2][0]-dL_dMt[0][2]) + 2*x*(dL_dMt[1][2]-dL_dMt[2][1]);
	dL_dq.y = 2*y*(dL_dMt[1][0]+dL_dMt[0][1]) + 2*z*(dL_dMt[2][0]+dL_dMt[0][2]) + 2*r*(dL_dMt[1][2]-dL_dMt[2][1]) - 4*x*(dL_dMt[2][2]+dL_dMt[1][1]);
	dL_dq.z = 2*x*(dL_dMt[1][0]+dL_dMt[0][1]) + 2*r*(dL_dMt[2][0]-dL_dMt[0][2]) + 2*z*(dL_dMt[1][2]+dL_dMt[2][1]) - 4*y*(dL_dMt[2][2]+dL_dMt[0][0]);
	dL_dq.w = 2*r*(dL_dMt[0][1]-dL_dMt[1][0]) + 2*x*(dL_dMt[2][0]+dL_dMt[0][2]) + 2*y*(dL_dMt[1][2]+dL_dMt[2][1]) - 4*z*(dL_dMt[1][1]+dL_dMt[0][0]);
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w};
}

// ============================================================
// Preprocess backward  (unchanged)
// ============================================================
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means, const int* radii,
	const float* shs, const bool* clamped,
	const glm::vec3* scales, const glm::vec4* rotations,
	const float scale_modifier, const float* proj,
	const glm::vec3* campos, const float3* dL_dmean2D,
	glm::vec3* dL_dmeans, float* dL_dcolor,
	float* dL_dcov3D, float* dL_dsh,
	glm::vec3* dL_dscale, glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0)) return;

	float3 m = means[idx];
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);
	glm::vec3 dL_dmean;
	float mul1 = (proj[0]*m.x + proj[4]*m.y + proj[8]*m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1]*m.x + proj[5]*m.y + proj[9]*m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0]*m_w - proj[3]*mul1)*dL_dmean2D[idx].x + (proj[1]*m_w - proj[3]*mul2)*dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4]*m_w - proj[7]*mul1)*dL_dmean2D[idx].x + (proj[5]*m_w - proj[7]*mul2)*dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8]*m_w - proj[11]*mul1)*dL_dmean2D[idx].x + (proj[9]*m_w - proj[11]*mul2)*dL_dmean2D[idx].y;
	dL_dmeans[idx] += dL_dmean;
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped,
		                   (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx],
		             dL_dcov3D, dL_dscale, dL_drot);
}

// ============================================================
// OPTIMIZED backward render kernel.
//
// Key optimisations over the baseline:
//   1. Tile-level early stopping via max_contrib_tile: we only
//      process rounds up to ceil(max_contrib / BLOCK_SIZE) so
//      tiles that saturate early skip many wasted outer rounds.
//   2. Snapshot-based T restoration: instead of starting with
//      T_final and un-multiplying through ALL skipped splats, we
//      read the pre-saved transmittance from the snapshot at the
//      bucket boundary just above max_contrib. This makes T
//      immediately correct, removing the need for
//      T = T / (1-alpha) on skipped splats.
//
// Gradient math is IDENTICAL to the original 3DGS backward;
// only the traversal range and initial T change.
// ============================================================
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const int W, int H,
	const float* means3D,
	const float* cam_pos,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ normals,
	const float* __restrict__ albedo,
	const float* __restrict__ roughness,
	const float* __restrict__ metallic,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	// --- SampleState ---
	const uint32_t* __restrict__ bucket_to_tile,
	const float*    __restrict__ sampled_T,
	const float*    __restrict__ sampled_ar,
	const uint32_t* __restrict__ max_contrib_tile,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	// -------------------
	const float* __restrict__ dL_dpixels_depth,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_opacity,
	const float* __restrict__ dL_dpixels_normal,
	const float* __restrict__ dL_dpixels_albedo,
	const float* __restrict__ dL_dpixels_roughness,
	const float* __restrict__ dL_dpixels_metallic,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_depth,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dnormals,
	float* __restrict__ dL_dalbedo,
	float* __restrict__ dL_droughness,
	float* __restrict__ dL_dmetallic)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	const uint2 range = ranges[tile_id];

	// ---- Optimisation 1: tile-level max_contrib ----
	const uint32_t tile_max = max_contrib_tile[tile_id];
	const int total_splats  = (int)(range.y - range.x);

	// Number of 32-splat buckets needed to cover all relevant splats
	const int top_bucket_ceil = (int)((tile_max + 31) / 32);          // ceil(tile_max / 32)
	const int num_buckets_in_tile = (total_splats + 31) / 32;
	const int effective_top = min(top_bucket_ceil * 32, total_splats); // clamped splat limit

	// We process backwards through [0, effective_top)
	const int rounds = (effective_top + BLOCK_SIZE - 1) / BLOCK_SIZE;

	bool done = !inside;
	int toDo = effective_top;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// ---- Optimisation 2: restore T from snapshot ----
	// T_forward at position effective_top = sampled_T at bucket top_bucket_ceil
	// (snapshot was written at the START of each bucket, so bucket top_bucket_ceil
	// holds the T value AFTER all splats in buckets 0..top_bucket_ceil-1).
	const uint32_t bucket_base = (tile_id == 0) ? 0u : per_tile_bucket_offset[tile_id - 1];
	const uint32_t global_top_bucket = bucket_base + (uint32_t)top_bucket_ceil;

	float T;
	if (inside) {
		if (top_bucket_ceil < num_buckets_in_tile) {
			// Read forward T at position effective_top from snapshot
			T = sampled_T[global_top_bucket * BLOCK_SIZE + block.thread_rank()];
		} else {
			// effective_top >= total_splats: use final_T
			T = final_Ts[pix_id];
		}
	} else {
		T = 0.0f;
	}

	// ---- Per-pixel backward state ----
	const float T_final = inside ? final_Ts[pix_id] : 0.0f;
	uint32_t contributor = (uint32_t)effective_top;
	const int last_contributor = inside ? (int)n_contrib[pix_id] : 0;

	float last_alpha = 0.0f;
	float accum_opacity = 0.0f;
	float accum_rec[C] = { 0.0f };
	float dL_dpixel[C], dL_dpixel_normal[C], dL_dpixel_albedo[C];
	float dL_dpixel_opacity, dL_dpixel_roughness, dL_dpixel_metallic, dL_dpixel_depth;

	if (inside) {
		for (int i = 0; i < C; i++) {
			dL_dpixel[i]        = dL_dpixels[i * H * W + pix_id];
			dL_dpixel_normal[i] = dL_dpixels_normal[i * H * W + pix_id];
			dL_dpixel_albedo[i] = dL_dpixels_albedo[i * H * W + pix_id];
		}
		dL_dpixel_opacity   = dL_dpixels_opacity[pix_id];
		dL_dpixel_roughness = dL_dpixels_roughness[pix_id];
		dL_dpixel_metallic  = dL_dpixels_metallic[pix_id];
		dL_dpixel_depth     = dL_dpixels_depth[pix_id];
	}
	float last_color[C] = { 0.0f };

	// Skip edge pixels' normal gradients
	if (pix.x == 0 || pix.x == W-1 || pix.y == 0 || pix.y == H-1)
		for (int i = 0; i < C; i++) dL_dpixel_normal[i] = 0.0f;

	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;

	// ---- Backward traversal (from effective_top back to 0) ----
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		// Load splats in reverse order, but starting from effective_top
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (progress < effective_top)
		{
			// Load from position: range.x + effective_top - progress - 1 (back-to-front)
			const int load_idx = (int)(range.x) + effective_top - progress - 1;
			const int coll_id  = point_list[load_idx];
			collected_id[block.thread_rank()]           = coll_id;
			collected_xy[block.thread_rank()]           = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int ch = 0; ch < C; ch++)
				collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ch];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			if (contributor >= (uint32_t)last_contributor)
				continue;

			const float2 xy    = collected_xy[j];
			const float2 d     = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power  = -0.5f * (con_o.x*d.x*d.x + con_o.z*d.y*d.y) - con_o.y*d.x*d.y;
			if (power > 0.0f) continue;

			const float G     = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f) continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			float3 view_dir = {
				cam_pos[0] - means3D[collected_id[j] * 3 + 0],
				cam_pos[1] - means3D[collected_id[j] * 3 + 1],
				cam_pos[2] - means3D[collected_id[j] * 3 + 2],
			};

			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				accum_rec[ch]   = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch]  = c;
				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				const float dL_dchannel_normal = dL_dpixel_normal[ch];
				atomicAdd(&(dL_dnormals[global_id * C + ch]), dchannel_dcolor * dL_dchannel_normal);
				const float dL_dchannel_albedo = dL_dpixel_albedo[ch];
				atomicAdd(&(dL_dalbedo[global_id * C + ch]), dchannel_dcolor * dL_dchannel_albedo);
			}
			atomicAdd(&(dL_droughness[global_id]), dchannel_dcolor * dL_dpixel_roughness);
			atomicAdd(&(dL_dmetallic[global_id]),  dchannel_dcolor * dL_dpixel_metallic);
			atomicAdd(&(dL_depth[global_id]),      dchannel_dcolor * dL_dpixel_depth);

			accum_opacity = last_alpha + (1.f - last_alpha) * accum_opacity;
			dL_dalpha += (1.0f - accum_opacity) * dL_dpixel_opacity;
			dL_dalpha *= T;
			last_alpha  = alpha;

			float bg_dot_dpixel = 0.0f;
			for (int ch = 0; ch < C; ch++)
				bg_dot_dpixel += bg_color[ch] * dL_dpixel[ch];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			const float dL_dG  = con_o.w * dL_dalpha;
			const float gdx    = G * d.x, gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
			const float abs_dL = abs(dL_dG*dG_ddelx*ddelx_dx) + abs(dL_dG*dG_ddely*ddely_dy);
			atomicAdd(&dL_dmean2D[global_id].z, abs_dL);

			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

// ============================================================
// Feature backward (unchanged)
// ============================================================
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderFeatureBackwardCUDA(
	const int W, int H,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ dL_dpixels_feature,
	const int feature_dim,
	float* __restrict__ dL_dfeature)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	float T = 1.0f;
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
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
			float power = -0.5f*(con_o.x*d.x*d.x + con_o.z*d.y*d.y) - con_o.y*d.x*d.y;
			if (power > 0.0f) continue;
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f/255.0f) continue;
			float test_T = T * (1-alpha);
			if (test_T < 0.0001f) { done = true; continue; }
			const float weight = alpha * T;
			const int point_offset = collected_id[j] * feature_dim;
			if (inside)
				for (int feat_ch = 0; feat_ch < feature_dim; ++feat_ch)
					atomicAdd(&(dL_dfeature[point_offset + feat_ch]),
					          weight * dL_dpixels_feature[feat_ch * H * W + pix_id]);
			T = test_T;
		}
	}
}

// ============================================================
// SSR backward (unchanged)
// ============================================================
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
SSRCUDA(
	int W, int H, const float focal_x, const float focal_y,
	const float* __restrict__ out_normal,
	const float* __restrict__ out_pos,
	const float* __restrict__ out_rgb,
    const float* __restrict__ out_albedo,
    const float* __restrict__ out_roughness,
    const float* __restrict__ out_metallic,
    const float* __restrict__ out_F0,
	const float* __restrict__ dL_dpixels,
	float* __restrict__ dl_albedo,
	float* __restrict__ dl_roughness,
	float* __restrict__ dl_metallic)
{
	auto block = cg::this_thread_block();
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
    dl_albedo[pix_id]           = 0.0f;
    dl_albedo[1*H*W + pix_id]   = 0.0f;
    dl_albedo[2*H*W + pix_id]   = 0.0f;
    dl_roughness[pix_id]        = 0.0f;
    dl_roughness[1*H*W + pix_id]= 0.0f;
    dl_roughness[2*H*W + pix_id]= 0.0f;
	dl_metallic[pix_id]         = 0.0f;
    dl_metallic[1*H*W + pix_id] = 0.0f;
    dl_metallic[2*H*W + pix_id] = 0.0f;
}

// ============================================================
// BACKWARD namespace wrappers
// ============================================================

void BACKWARD::render_feature(
	const dim3 grid, const dim3 block,
	const int W, int H,
	const uint2* ranges, const uint32_t* point_list,
	const float2* means2D, const float4* conic_opacity,
	const float* dL_dpixels_feature, const int feature_dim,
	float* dL_dfeature)
{
	renderFeatureBackwardCUDA<<<grid, block>>>(W, H, ranges, point_list,
		means2D, conic_opacity, dL_dpixels_feature, feature_dim, dL_dfeature);
}

void BACKWARD::preprocess(
	const int P, int D, int M,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float3* means3D, const int* radii,
	const float* shs, const bool* clamped,
	const glm::vec3* scales, const glm::vec4* rotations,
	const float scale_modifier, const float* cov3Ds,
	const float* viewmatrix, const float* projmatrix,
	const glm::vec3* campos,
	const float3* dL_dmean2D, const float* dL_dconic,
	const float* dL_depth,
	glm::vec3* dL_dmean3D, float* dL_dcolor,
	float* dL_dcov3D, float* dL_dsh,
	glm::vec3* dL_dscale, glm::vec4* dL_drot)
{
	computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
		P, means3D, radii, cov3Ds,
		focal_x, focal_y, tan_fovx, tan_fovy,
		viewmatrix, dL_dconic, dL_depth,
		(float3*)dL_dmean3D, dL_dcov3D);

	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M, (float3*)means3D, radii, shs, clamped,
		(glm::vec3*)scales, (glm::vec4*)rotations, scale_modifier,
		projmatrix, campos, (float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D, dL_dcolor, dL_dcov3D,
		dL_dsh, dL_dscale, dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const int W, int H,
	const float* means3D, const float* cam_pos,
	const uint2* ranges, const uint32_t* point_list,
	const float* bg_color,
	const float2* means2D, const float4* conic_opacity,
	const float* colors, const float* normal,
	const float* albedo, const float* roughness,
	const float* metallic,
	const float* final_Ts, const uint32_t* n_contrib,
	// --- SampleState ---
	const uint32_t* bucket_to_tile,
	const float*    sampled_T,
	const float*    sampled_ar,
	const uint32_t* max_contrib_tile,
	const uint32_t* per_tile_bucket_offset,
	// -------------------
	const float* dL_dpixels_depth,
	const float* dL_dpixels,
	const float* dL_dpixels_opacity,
	const float* dL_dpixels_normal,
	const float* dL_dpixels_albedo,
	const float* dL_dpixels_roughness,
	const float* dL_dpixels_metallic,
	float3* dL_dmean2D, float4* dL_dconic2D,
	float* dL_depth, float* dL_dopacity,
	float* dL_dcolors, float* dL_dnormals,
	float* dL_dalbedo, float* dL_droughness,
	float* dL_dmetallic)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		W, H, means3D, cam_pos,
		ranges, point_list, bg_color,
		means2D, conic_opacity,
		colors, normal, albedo, roughness, metallic,
		final_Ts, n_contrib,
		bucket_to_tile, sampled_T, sampled_ar,
		max_contrib_tile, per_tile_bucket_offset,
		dL_dpixels_depth, dL_dpixels, dL_dpixels_opacity,
		dL_dpixels_normal, dL_dpixels_albedo,
		dL_dpixels_roughness, dL_dpixels_metallic,
		dL_dmean2D, dL_dconic2D,
		dL_depth, dL_dopacity, dL_dcolors,
		dL_dnormals, dL_dalbedo, dL_droughness, dL_dmetallic);
}

void BACKWARD::SSR(
	const dim3 grid, const dim3 block,
	int W, int H, const float focal_x, const float focal_y,
	const float* out_normal, const float* out_pos,
	const float* out_rgb, const float* out_albedo,
    const float* out_roughness, const float* out_metallic,
    const float* out_F0, const float* dL_dpixels,
	float* dl_albedo, float* dl_roughness, float* dl_metallic)
{
	SSRCUDA<<<grid, block>>>(W, H, focal_x, focal_y,
		out_normal, out_pos, out_rgb, out_albedo,
		out_roughness, out_metallic, out_F0, dL_dpixels,
		dl_albedo, dl_roughness, dl_metallic);
}
