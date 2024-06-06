#include <torch/extension.h>

// Return indices
torch::Tensor euclidDistanceIndexedCUDA(
    const torch::Tensor& data,
    const torch::Tensor& clusters
);

// Return amount of vertices mapped
int euclidDistanceMappedCUDA(
    int sampleSetOneId,
    const torch::Tensor& sampleSetOne,
    const torch::Tensor& sampleSetTwo,
    torch::Tensor& sampleMapOne,
    torch::Tensor& sampleMapTwo,
    torch::Tensor& firstSampleMap // To merge graphs into a degenerate graph
);

// Return a list of clusters
std::vector<torch::Tensor> collectGroups(
    int sampleSize,
    int numClusters,
    const torch::Tensor& sampleSets,
    const torch::Tensor& sampleMaps
);

// Returns the indice of the medoid
int findMedoids(
    const torch::Tensor& cluster
);