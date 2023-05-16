#include "gather.h"
#include "scatter.h"
#include "scatter_gather.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Sparse Incremental Generative Engine (SIGE)";
    m.def("gather", &gather_mps, "Gather (MPS)");
    m.def("scatter", &scatter_mps, "Scatter (MPS)");
    m.def("scatter_with_block_residual", &scatter_with_block_residual_mps, "Scatter with block residual (MPS)");
    m.def("scatter_gather", &scatter_gather_mps, "Scatter-Gather (MPS)");
    m.def("get_scatter_map", &get_scatter_map_mps, "Get scatter map (MPS)");
}
