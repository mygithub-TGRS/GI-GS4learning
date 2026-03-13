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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		float3* pos_view;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	// ---------------------------------------------------------------
	// SampleState: per-bucket snapshots for optimized backward pass.
	//
	// Layout:
	//   bucket_to_tile[b]                            -> tile id of bucket b
	//   sampled_T[b * BLOCK_SIZE + pix]              -> transmittance T
	//                                                   at the START of bucket b
	//                                                   for pixel pix in its tile
	//   sampled_ar[b * BLOCK_SIZE * C + ch*BLOCK_SIZE + pix]
	//                                                -> accumulated color channel ch
	//                                                   at the START of bucket b
	//   max_contrib[tile_id]                         -> maximum n_contrib across
	//                                                   all pixels in the tile
	//   per_tile_bucket_count[tile_id]               -> number of 32-splat buckets
	//   per_tile_bucket_offset[tile_id]              -> exclusive prefix sum of above
	// ---------------------------------------------------------------
	struct SampleState
	{
		// Per-bucket arrays (size = num_buckets)
		uint32_t* bucket_to_tile;

		// Per-bucket-per-pixel arrays
		// BLOCK_SIZE = BLOCK_X * BLOCK_Y = 256 pixels per tile
		float* sampled_T;           // [num_buckets * BLOCK_SIZE]
		float* sampled_ar;          // [C * num_buckets * BLOCK_SIZE]

		// Per-tile arrays (size = num_tiles)
		uint32_t* max_contrib;
		uint32_t* per_tile_bucket_count;
		uint32_t* per_tile_bucket_offset;

		static SampleState fromChunk(char*& chunk, size_t num_tiles, size_t num_buckets, size_t C);
	};

	template<typename T>
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	// Specialised helper for SampleState which needs three size arguments.
	inline size_t requiredSampleState(size_t num_tiles, size_t num_buckets, size_t C)
	{
		char* size = nullptr;
		SampleState::fromChunk(size, num_tiles, num_buckets, C);
		return ((size_t)size) + 128;
	}
};
