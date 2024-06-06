#include "euclidDistance.h"
#include <float.h>
#include <cmath>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

/**************************************************SHARED********************************************************************/

__host__ __device__ float euclidDistance(const int dimSize, const float *data, const float *cluster) {
  float result = 0.0;
  for (int i = 0; i < dimSize; i++) {
    float diff = data[i] - cluster[i];
    result += diff * diff;
  }

  return result;
}

// Multiple atomic swap solution:
// https://stackoverflow.com/questions/17411493/how-can-i-implement-a-custom-atomic-function-involving-several-variables 
typedef union {
  float floats[2];                 // floats[0] = lowest
  int ints[2];                     // ints[1] = lowIdx
  unsigned long long int ulong;    // for atomic update
} myAtomics;

__device__ unsigned long long int myAtomicMin(unsigned long long int* address, float val1, int val2) {
    myAtomics loc, loctest;
    loc.floats[0] = val1;
    loc.ints[1] = val2;
    loctest.ulong = *address;
    while (loctest.floats[0] >  val1) { // Data can only get smaller -> no race condition possible
      loctest.ulong = atomicCAS(address, loctest.ulong, loc.ulong); // Reassign in case of race condition, if success assignment loc.ulong -> extra iteration needed to break out of loop
    }
    return loctest.ulong;
}

/**************************************************KERNELS********************************************************************/

__global__ void euclidDistanceIndexed(
  int dataSize, int clusterSize, int dimSize,
  const float *data,
  const float *clusters,
  int64_t *indices
) {

  auto idx = cg::this_grid().thread_rank();
  if (idx >= dataSize)
    return;

  float minDistance = FLT_MAX;
  int64_t minIndex;
  for (int64_t i = 0; i < clusterSize; i++) {
    float d = euclidDistance(dimSize, &(data[idx * dimSize]), &(clusters[i * dimSize]));
    if (d < minDistance) {
      minDistance = d; minIndex = i;
    }
  }

  indices[idx] = minIndex;
}

__global__ void euclidDistanceMapped(
  int dataSize, int dimSize,
  int sampleSetOneId,
  const float *sampleSetOne,
  const float *sampleSetTwo,
  int *sampleMapOne,
  int *sampleMapTwo,
  int *firstSampleMap,
  int *verticesMapped
) {

  auto idx = cg::this_grid().thread_rank();
  // Vertice is not mapped
  if (idx >= dataSize || sampleMapOne[idx * 2 + 1] == -2)
    return;

  float minDistance = FLT_MAX;
  int64_t minIndex;
  for (int64_t i = 0; i < dataSize; i++) {
    float d = euclidDistance(dimSize, &(sampleSetOne[idx * dimSize]), &(sampleSetTwo[i * dimSize]));
    if (d < minDistance) {
      minDistance = d; minIndex = i;
    }
  }

  // Exchange tail first to avoid race conditions - see initSampleSets for "if" explanation
  auto old = atomicExch(&(sampleMapTwo[minIndex * 2 + 1]), sampleMapOne[idx * 2 + 1]);
  if (old == -2) {
    sampleMapTwo[minIndex * 2] = dataSize * sampleSetOneId + idx;
    atomicAdd(verticesMapped, 1);
  }
  else {
    firstSampleMap[old * 2] = dataSize * sampleSetOneId + idx;
  }
}

__global__ void euclidDistanceMinimal(
  int dataSize, int dimSize,
  const float *data,
  myAtomics *minPoint
) {

  auto idx = cg::this_grid().thread_rank();
  if (idx >= dataSize)
    return;

  float totalDistance = 0.0;
  for (int64_t i = 0; i < dataSize; i++) {
    totalDistance += euclidDistance(dimSize, &(data[idx * dimSize]), &(data[i * dimSize]));;
  }

  myAtomicMin(&(minPoint->ulong), totalDistance, (int)idx);
}

/**************************************************C++********************************************************************/

torch::Tensor euclidDistanceIndexedCUDA(const torch::Tensor &data, const torch::Tensor &clusters) {
  if (data.ndimension() != 2 || clusters.ndimension() != 2) {
    AT_ERROR("Data and clusters must have dimension 2");
  }

  if (data.size(1) != clusters.size(1)) {
    AT_ERROR("Data and clusters must have same number of channels");
  }

  torch::Tensor indices = torch::empty({data.size(0)}, data.options().dtype(torch::kInt64));
  int dataSize = data.size(0);
  int dimSize = data.size(1);
  int clusterSize = clusters.size(0);

  // +31 for rounding
  euclidDistanceIndexed<<<(dataSize + 31) / 32, 32>>>(
      dataSize, clusterSize, dimSize,
      data.contiguous().data_ptr<float>(),
      clusters.contiguous().data_ptr<float>(),
      indices.contiguous().data_ptr<int64_t>()
  );

  return indices;
}

int euclidDistanceMappedCUDA(
    int sampleSetOneId,
    const torch::Tensor& sampleSetOne,
    const torch::Tensor& sampleSetTwo,
    torch::Tensor& sampleMapOne,
    torch::Tensor& sampleMapTwo,
    torch::Tensor& firstSampleMap
) {
  if (sampleSetOne.ndimension() != 2 || sampleSetTwo.ndimension() != 2) {
    AT_ERROR("Sample sets must have dimension 2");
  }

  if (sampleSetOne.size(0) != sampleSetTwo.size(0)) {
    AT_ERROR("Sample sets must have the same cardinality");
  }

  if (sampleSetOne.size(1) != sampleSetTwo.size(1)) {
    AT_ERROR("Sample sets must have the same number of channels");
  }

  int *verticesMapped;
  int zero = 0;
  cudaMalloc(&verticesMapped, sizeof(int));
  cudaMemcpy(verticesMapped, &zero, sizeof(int), cudaMemcpyHostToDevice);
  int dataSize = sampleSetOne.size(0);
  int dimSize = sampleSetOne.size(1);

  // +31 for rounding
  euclidDistanceMapped<<<(dataSize + 31) / 32, 32>>>(
      dataSize, dimSize,
      sampleSetOneId,
      sampleSetOne.contiguous().data_ptr<float>(),
      sampleSetTwo.contiguous().data_ptr<float>(),
      sampleMapOne.contiguous().data_ptr<int>(),
      sampleMapTwo.contiguous().data_ptr<int>(),
      firstSampleMap.contiguous().data_ptr<int>(),
      verticesMapped
  );
  cudaDeviceSynchronize();


  int copy;
  cudaMemcpy(&copy, verticesMapped, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(verticesMapped);

  return copy;
}

std::vector<torch::Tensor> collectGroups(
    int sampleSize,
    int numClusters,
    const torch::Tensor& sampleSets,
    const torch::Tensor& sampleMaps
) {
  std::vector<torch::Tensor> ret{};
  std::vector<torch::Tensor> val{};

  int numSampleSets = sampleSets.size(0);
  for (int i = 0; i < sampleMaps.size(1); ++i) {
    int next = sampleMaps[numSampleSets - 1][i][0].item<int>();
    if (next == -2) {
      continue;
    }
    val.push_back(sampleSets[numSampleSets - 1][i]);
    while (next != -1) {
      int setId = floor(next / sampleSize);
      int offset = next % sampleSize;
      val.push_back(sampleSets[setId][offset]);
      next = sampleMaps[setId][offset][0].item<int>();
    }

    ret.push_back(torch::stack(val));
    val.clear();

    // Because of margin, take first n clusters
    if (ret.size() == numClusters) {
      return ret;
    }
  }

  return ret;
}

int findMedoids(
    const torch::Tensor& cluster
) {
  myAtomics *minPoint;
  float max = FLT_MAX;
  cudaMalloc(&minPoint, sizeof(myAtomics));
  cudaMemcpy(&(minPoint->floats[0]), &max, sizeof(myAtomics), cudaMemcpyHostToDevice);

  int dataSize = cluster.size(0);
  int dimSize = cluster.size(1);

  // +31 for rounding
  euclidDistanceMinimal<<<(dataSize + 31) / 32, 32>>>(
      dataSize, dimSize,
      cluster.contiguous().data_ptr<float>(),
      minPoint
  );
  cudaDeviceSynchronize();

  int copy;
  cudaMemcpy(&copy, &(minPoint->ints[1]), sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(minPoint);

  return copy;

}

// Modified from "Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis":
// https://github.com/KeKsBoTer/c3dgs/blob/master/submodules/weighted_distance/weighted_distance.cu