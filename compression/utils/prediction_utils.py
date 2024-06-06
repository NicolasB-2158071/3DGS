import torch
import time

from gaussian_model import GaussianModel
def predictTotalTime(
        device: torch.device,
        gaussians: GaussianModel,
        DCSHIterations: int,
        DCSHClusters: int,
        ScRoIterations: int,
        ScRoClusters: int
    ) -> None:
    sum: float = 0.0
    sum += predictTimeNeeded(gaussians.get_features_flat, device, numIterations=DCSHIterations, numClusters=DCSHClusters)
    sum += predictTimeNeeded(gaussians._scaling, device, numIterations=ScRoIterations, numClusters=ScRoClusters)
    sum += predictTimeNeeded(gaussians._rotation, device, numIterations=ScRoIterations, numClusters=ScRoClusters)
    print("Training time needed (min):", sum)

def predictTimeNeeded(data: torch.Tensor, device: torch.device, numIterations: int = 3000, batchSize: int = 100000, numClusters: int = 2 ** 12) -> float:
    from compression import Kmeans
    kmeans: Kmeans = Kmeans(data, device=device)
    kmeans.initClusters(numClusters=numClusters)
    torch.cuda.synchronize()

    t0 = time.time()
    kmeans.runIterations(numIterations=1, batchSize=batchSize)
    torch.cuda.synchronize()
    t1 = time.time() - t0
    
    iterationsPerSecond = 1 / t1
    trainingSeconds = numIterations / iterationsPerSecond
    return trainingSeconds / 60

def predictAverageAccuracy(originalData: torch.Tensor, codeBook: torch.Tensor, indices: torch.Tensor) -> None:
    sum: torch.Tensor = torch.zeros(1, originalData.shape[1]).to("cpu")
    for i in range(originalData.shape[0]):
        sum += (originalData[i] - codeBook[indices[i].item()] )
    print("Average accuracy:", sum / originalData.shape[0])