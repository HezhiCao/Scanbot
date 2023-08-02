#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void construct_quality_and_semantic_channels_cuda(
    torch::Tensor semantic_map_points, torch::Tensor topdown_map,
    const std::vector<long>& agent_position, unsigned quality_channel_begin, float map_cell_size,
    float far_threshold, float best_scanning_distance, float orientation_division_interval,
    torch::Tensor semantic_channel_labels, torch::Tensor category_mapping,
    unsigned semantic_channel_begin, bool has_others_channel);

void construct_all_channels_cuda(torch::Tensor semantic_local_points, torch::Tensor topdown_map,
                                 torch::Tensor current_pose, unsigned obstacle_channel,
                                 unsigned exploration_channel, unsigned quality_channel_begin,
                                 unsigned semantic_channel_begin, float map_cell_size,
                                 unsigned map_size_in_cells, float near_threshold,
                                 float far_threshold, float low_threshold,
                                 float best_scanning_distance, float orientation_division_interval,
                                 torch::Tensor semantic_channel_labels,
                                 torch::Tensor category_mapping, bool has_others_channel);

void construct_quality_and_semantic_channels(
    torch::Tensor semantic_map_points, torch::Tensor topdown_map,
    const std::vector<long>& agent_position, unsigned quality_channel_begin, float map_cell_size,
    float far_threshold, float best_scanning_distance, float orientation_division_interval,
    torch::Tensor semantic_channel_labels, torch::Tensor category_mapping,
    unsigned semantic_channel_begin, bool has_others_channel) {
    CHECK_INPUT(semantic_map_points);
    CHECK_INPUT(topdown_map);
    CHECK_INPUT(semantic_channel_labels);
    CHECK_INPUT(category_mapping);

    construct_quality_and_semantic_channels_cuda(
        semantic_map_points, topdown_map, agent_position, quality_channel_begin, map_cell_size,
        far_threshold, best_scanning_distance, orientation_division_interval,
        semantic_channel_labels, category_mapping, semantic_channel_begin, has_others_channel);
}

void construct_all_channels(torch::Tensor semantic_local_points, torch::Tensor topdown_map,
                            torch::Tensor current_pose, unsigned obstacle_channel,
                            unsigned exploration_channel, unsigned quality_channel_begin,
                            unsigned semantic_channel_begin, float map_cell_size,
                            unsigned map_size_in_cells, float near_threshold, float far_threshold,
                            float low_threshold, float best_scanning_distance,
                            float orientation_division_interval,
                            torch::Tensor semantic_channel_labels, torch::Tensor category_mapping,
                            bool has_others_channel) {
    CHECK_INPUT(semantic_local_points);
    CHECK_INPUT(topdown_map);
    CHECK_INPUT(current_pose);
    CHECK_INPUT(semantic_channel_labels);
    CHECK_INPUT(category_mapping);

    construct_all_channels_cuda(semantic_local_points, topdown_map, current_pose, obstacle_channel,
                                exploration_channel, quality_channel_begin, semantic_channel_begin,
                                map_cell_size, map_size_in_cells, near_threshold, far_threshold,
                                low_threshold, best_scanning_distance,
                                orientation_division_interval, semantic_channel_labels,
                                category_mapping, has_others_channel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("construct_quality_and_semantic_channels", &construct_quality_and_semantic_channels,
          "Construct quality & semantic channels (CUDA)");
    m.def("construct_all_channels", &construct_all_channels,
          "Construct obstacle, exploration, quality and semantic channels (CUDA)");
}
