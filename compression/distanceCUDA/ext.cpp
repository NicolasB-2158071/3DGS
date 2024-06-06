#include <torch/extension.h>
#include "euclidDistance.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("euclidDistanceIndexed", &euclidDistanceIndexedCUDA);
    m.def("euclidDistanceMapped", &euclidDistanceMappedCUDA);
    m.def("collectGroups", &collectGroups);
    m.def("findMedoids", &findMedoids);
}