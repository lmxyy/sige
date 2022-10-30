#include "gather.cpp"
#include "scatter.cpp"
#include "scatter_gather.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Sparse Incremental Generative Engine (SIGE)";
    m.def("gather", &gather_cuda, "Gather (CUDA)");
    m.def("scatter", &scatter_cuda, "Scatter (CUDA)");
    m.def("scatter_with_block_residual", &scatter_with_block_residual_cuda, "Scatter with block residual (CUDA)");
    m.def("scatter_gather", &scatter_gather_cuda, "Scatter-Gather (CUDA)");
    m.def("get_scatter_map", &get_scatter_map_cuda, "Get scatter map (CUDA)");
}