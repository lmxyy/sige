#include "gather.cpp"
#include "scatter.cpp"
#include "scatter_gather.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Sparse Incremental Generative Engine (SIGE)";
    m.def("gather", &gather_cpu, "Gather (CPU)");
    m.def("scatter", &scatter_cpu, "Scatter (CPU)");
    m.def("scatter_with_block_residual", &scatter_with_block_residual_cpu, "Scatter with block residual (CPU)");
    m.def("scatter_gather", &scatter_gather_cpu, "Scatter-Gather (CPU)");
    m.def("get_scatter_map", &get_scatter_map_cpu, "Get scatter map (CPU)");
}
