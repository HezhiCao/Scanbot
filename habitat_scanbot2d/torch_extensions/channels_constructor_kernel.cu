#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <vector>

#define MAX_NUM_LABELS 20
#define MAX_NUM_INSTANCES 4096
__constant__ int CHANNEL_LABELS[MAX_NUM_LABELS];
__constant__ float CURRENT_POSE[12];

__device__ __forceinline__ float3 transfrom_local_to_global(float3 local_point) {
    return make_float3(CURRENT_POSE[0] * local_point.x + CURRENT_POSE[1] * local_point.y +
                           CURRENT_POSE[2] * local_point.z + CURRENT_POSE[3],
                       CURRENT_POSE[4] * local_point.x + CURRENT_POSE[5] * local_point.y +
                           CURRENT_POSE[6] * local_point.z + CURRENT_POSE[7],
                       CURRENT_POSE[8] * local_point.x + CURRENT_POSE[9] * local_point.y +
                           CURRENT_POSE[10] * local_point.z + CURRENT_POSE[11]);
}

__device__ __forceinline__ int2 transform_global_to_map(float3 global_point, float map_cell_size,
                                                        unsigned map_size_in_cells) {
    int translation = map_size_in_cells / 2;
    int raw = global_point.z / map_cell_size + translation;
    int col = global_point.x / map_cell_size + translation;
    raw = min(max(raw, 0), map_size_in_cells - 1);
    col = min(max(col, 0), map_size_in_cells - 1);
    return make_int2(raw, col);
}

__device__ __forceinline__ float compute_distance_score(int2 agent_position, int2 map_point,
                                                        float map_cell_size, float far_threshold,
                                                        float best_scanning_distance) {
    float scanning_distance =
        sqrtf((agent_position.x - map_point.x) * (agent_position.x - map_point.x) +
              (agent_position.y - map_point.y) * (agent_position.y - map_point.y)) *
        map_cell_size;
    float max_difference = fmaxf(far_threshold - best_scanning_distance, best_scanning_distance);
    float bin_length = max_difference / 5;
    float distance_score = (max_difference - fabsf(scanning_distance - best_scanning_distance));
    distance_score = ceilf(distance_score / bin_length) / 5;
    return distance_score;
}

__device__ __forceinline__ unsigned compute_orientation_division(
    int2 agent_position, int2 map_point, float orientation_division_interval) {
    float angle = atan2f(map_point.x - agent_position.x, agent_position.y - map_point.y);
    angle = fmodf(angle + 2 * CUDART_PI_F + orientation_division_interval / 2, 2 * CUDART_PI_F);
    return angle / orientation_division_interval;
}

// This function is only used for optimization, please refer to
// SemanticTopDownCudaSensor._construct_quality_channels and
// SemanticTopDownCudaSensor._construct_semantic_channels for computational logic
// semantic_map_points is (n, 3) tensor, with 3 channels corresponding to (row, col, label)
__global__ void construct_quality_and_semantic_channels_kernel(
    const torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> semantic_map_points,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> topdown_map, long agent_row,
    long agent_col, unsigned quality_channel_begin, float map_cell_size, float far_threshold,
    float best_scanning_distance, float orientation_division_interval,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> semantic_channel_labels,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> category_mapping,
    unsigned semantic_channel_begin, bool has_others_channel) {
    __shared__ int channel_labels[MAX_NUM_LABELS];
    __shared__ int categories[MAX_NUM_INSTANCES];
    const unsigned num_semantic_channels = semantic_channel_labels.size(0);
    const unsigned num_instances = category_mapping.size(0);
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned label_id = threadIdx.x; label_id < num_semantic_channels;
         label_id += blockDim.x) {
        channel_labels[label_id] = semantic_channel_labels[label_id];
    }
    for (unsigned category_id = threadIdx.x; category_id < num_instances;
         category_id += blockDim.x) {
        categories[category_id] = category_mapping[category_id];
    }
    __syncthreads();

    if (point_index < semantic_map_points.size(0)) {
        const int label = categories[semantic_map_points[point_index][2]];
        // in gibson, label 0 means non-labeled
        // in mp3d, label 0 is void
        if (label > 0) {
            int2 agent_position = make_int2(agent_row, agent_col);
            int2 map_point =
                make_int2(semantic_map_points[point_index][0], semantic_map_points[point_index][1]);
            // Cosntruct quality channels
            if (orientation_division_interval > 0) {
                float distance_score =
                    compute_distance_score(agent_position, map_point, map_cell_size, far_threshold,
                                           best_scanning_distance);

                unsigned orientation_division = compute_orientation_division(
                    agent_position, map_point, orientation_division_interval);

                topdown_map[map_point.x][map_point.y]
                           [quality_channel_begin + orientation_division] = distance_score;
            }

            // Construct semantic channels
            bool belong_others = true;
            for (unsigned i = 0; i < num_semantic_channels; ++i) {
                // Label -1 means a nonexist category in current scene
                if (channel_labels[i] == label) {
                    topdown_map[map_point.x][map_point.y][semantic_channel_begin + i] = 1.0f;
                    belong_others = false;
                    break;
                }
            }

            // "others" always occupies last channel
            if (has_others_channel && belong_others) {
                topdown_map[map_point.x][map_point.y]
                           [semantic_channel_begin + num_semantic_channels] = 1.0f;
            }
        }
    }
}

__global__ void construct_all_channels_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> semantic_local_points,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> topdown_map,
    unsigned obstacle_channel, unsigned exploration_channel, unsigned quality_channel_begin,
    unsigned semantic_channel_begin, float map_cell_size, unsigned map_size_in_cells,
    float near_threshold, float far_threshold, float low_threshold, float best_scanning_distance,
    float orientation_division_interval,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> category_mapping,
    unsigned num_semantic_channels, bool has_others_channel) {
    extern __shared__ int categories[];
    const unsigned num_instances = category_mapping.size(0);
    for (unsigned instance_id = threadIdx.x; instance_id < num_instances;
         instance_id += blockDim.x) {
        categories[instance_id] = category_mapping[instance_id];
    }
    __syncthreads();

    unsigned point_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (point_index < semantic_local_points.size(0)) {
        float3 local_point = make_float3(semantic_local_points[point_index][0],
                                         semantic_local_points[point_index][1],
                                         semantic_local_points[point_index][2]);
        const int label = categories[static_cast<unsigned>(semantic_local_points[point_index][3])];
        if (abs(local_point.z) < far_threshold && abs(local_point.z) > near_threshold) {
            float3 global_point = transfrom_local_to_global(local_point);
            int2 map_point = transform_global_to_map(global_point, map_cell_size, map_size_in_cells);

            // construct exploration channel
            topdown_map[map_point.x][map_point.y][exploration_channel] = 1.0f;

            if (global_point.y < CURRENT_POSE[7] && global_point.y > low_threshold) {
                int2 agent_position = transform_global_to_map(
                    make_float3(CURRENT_POSE[3], CURRENT_POSE[7], CURRENT_POSE[11]), map_cell_size,
                    map_size_in_cells);
                // in gibson, label 0 means non-labeled
                // in mp3d, label 0 is void
                if (label > 0) {
                    // Cosntruct quality channels
                    if (orientation_division_interval > 0) {
                        float distance_score =
                            compute_distance_score(agent_position, map_point, map_cell_size,
                                                   far_threshold, best_scanning_distance);

                        unsigned orientation_division = compute_orientation_division(
                            agent_position, map_point, orientation_division_interval);

                        topdown_map[map_point.x][map_point.y]
                                   [quality_channel_begin + orientation_division] = distance_score;
                    }

                    // Construct semantic channels
                    bool belong_others = true;
                    for (unsigned i = 0; i < num_semantic_channels; ++i) {
                        // Label -1 means a nonexist category in current scene
                        if (CHANNEL_LABELS[i] == label) {
                            topdown_map[map_point.x][map_point.y][semantic_channel_begin + i] = 1.0f;
                            belong_others = false;
                            break;
                        }
                    }

                    // "others" always occupies last channel
                    if (has_others_channel && belong_others) {
                        topdown_map[map_point.x][map_point.y]
                                   [semantic_channel_begin + num_semantic_channels] = 1.0f;
                    }
                }

                // construct obstacle channel
                atomicAdd(&topdown_map[map_point.x][map_point.y][obstacle_channel], 1.0f);
            }
        }
    }
}

void construct_quality_and_semantic_channels_cuda(
    torch::Tensor semantic_map_points, torch::Tensor topdown_map,
    const std::vector<long>& agent_position, unsigned quality_channel_begin, float map_cell_size,
    float far_threshold, float best_scanning_distance, float orientation_division_interval,
    torch::Tensor semantic_channel_labels, torch::Tensor category_mapping,
    unsigned semantic_channel_begin, bool has_others_channel) {
    const unsigned threads = 1024;
    const unsigned blocks = (semantic_map_points.size(0) + threads - 1) / threads;

    construct_quality_and_semantic_channels_kernel<<<blocks, threads>>>(
        semantic_map_points.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
        topdown_map.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), agent_position[0],
        agent_position[1], quality_channel_begin, map_cell_size, far_threshold,
        best_scanning_distance, orientation_division_interval,
        semantic_channel_labels.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        category_mapping.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        semantic_channel_begin, has_others_channel);
}

void construct_all_channels_cuda(torch::Tensor semantic_local_points, torch::Tensor topdown_map,
                                 torch::Tensor current_pose, unsigned obstacle_channel,
                                 unsigned exploration_channel, unsigned quality_channel_begin,
                                 unsigned semantic_channel_begin, float map_cell_size,
                                 unsigned map_size_in_cells, float near_threshold,
                                 float far_threshold, float low_threshold,
                                 float best_scanning_distance, float orientation_division_interval,
                                 torch::Tensor semantic_channel_labels,
                                 torch::Tensor category_mapping, bool has_others_channel) {
    const auto num_points = semantic_local_points.size(0);
    const dim3 block = at::cuda::getApplyBlock();
    dim3 grid;
    at::cuda::getApplyGrid(num_points, grid, at::cuda::current_device());

    const unsigned num_semantic_channels = semantic_channel_labels.size(0);
    cudaMemcpyToSymbol(CHANNEL_LABELS, semantic_channel_labels.data<int>(),
                       num_semantic_channels * sizeof(int), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(CURRENT_POSE, current_pose.data<float>(), 12 * sizeof(float), 0,
                       cudaMemcpyDeviceToDevice);

    unsigned shared_memory = category_mapping.size(0) * sizeof(int);

    construct_all_channels_kernel<<<grid, block, shared_memory, at::cuda::getCurrentCUDAStream()>>>(
        semantic_local_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        topdown_map.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), obstacle_channel,
        exploration_channel, quality_channel_begin, semantic_channel_begin, map_cell_size,
        map_size_in_cells, near_threshold, far_threshold, low_threshold, best_scanning_distance,
        orientation_division_interval,
        category_mapping.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        num_semantic_channels, has_others_channel);
}
