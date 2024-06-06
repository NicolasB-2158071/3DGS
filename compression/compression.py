from argparse import ArgumentParser
from gaussian_model import GaussianModel
import torch
import numpy as np
from torch_scatter import scatter
from utils.prediction_utils import *
from euclidDistance import euclidDistanceIndexed, euclidDistanceMapped, collectGroups, findMedoids

import time

class Kmeans:
    class InitializeMethods:
        def initClustersRandom(self, data: torch.Tensor, numClusters: int) -> torch.Tensor:
            return data[np.random.choice(
                data.shape[0],
                numClusters,
                replace=False
            )]
        
        def initClusterspp(self, data: torch.Tensor, numClusters: int) -> torch.Tensor:
            # dist calculation needs to be faster to assign 4096 clusters,
            # even only calculating dist to the newly added cluster is to slow.
            pass

    def __init__(
            self,
            data: torch.Tensor,
            device: torch.device = "cpu"
        ):
        self._device: torch.device = device
        self._initMethods: Kmeans.InitializeMethods = Kmeans.InitializeMethods()

        self._data: torch.Tensor = data
        self._indices: torch.Tensor = torch.empty(self._data.shape[0], dtype=torch.int64).to(device=self._device)

    def initClusters(self, numClusters: int, initMethod: str = "random") -> None:
        match initMethod:
            case "random":
                self._clusters: torch.Tensor = self._initMethods.initClustersRandom(self._data, numClusters).to(self._device)
            case "pp":
                self._clusters: torch.Tensor = self._initMethods.initClusterspp(self._data, numClusters).to(self._device)
        self.assignAllClusterIndices() # Point all indices to something

    def assignAllClusterIndices(self) -> None:
        # Chunking solution to memory problem: Compact3D
        offset: int = 10000
        i: int = 0
        while True:
            dist: torch.Tensor = torch.cdist(self._data[i * offset : (i + 1) * offset, : ], self._clusters, p=2) # L^2 norm
            self._indices[i * offset : (i + 1) * offset] = torch.argmin(dist, dim=1)

            i += 1
            if i * offset > self._data.shape[0]:
                break
    
    def assignClusterIndicesBatched(self, batchSize: int) -> None:
        choice = np.random.choice(self._data.shape[0], batchSize, replace=False)
        self._indices[choice] = euclidDistanceIndexed(self._data[choice, : ], self._clusters)

    def meanClusters(self) -> None:
        scatter(self._data, self._indices, dim=0, reduce="mean", out=self._clusters, dim_size=self._clusters.shape[0])
        
    def runIterations(self, numIterations: int, batchSize: int = 2 ** 18) -> tuple:
        with torch.no_grad():
            for i in range(numIterations):
                self.meanClusters()
                self.assignClusterIndicesBatched(batchSize)

                if i % 500 == 0:
                    print("Done", i, "iterations")

            return self._clusters, self._indices

class AGORAS:
    def __init__(
            self,
            data: torch.Tensor,
            numClusters: int,
            numSampleSets: int,
            device: torch.device = "cpu"
        ):
        self._device: torch.device = device
        self._data: torch.Tensor = data.detach()
        self._indices: torch.Tensor = torch.empty(self._data.shape[0], dtype=torch.int64).to(device=self._device)

        self._numSampleSets: int = numSampleSets
        self._numClusters: int = numClusters

        self._GAMMA: float = 0.577215665
        self._MARGIN: int = np.ceil(self._numClusters / 40)

    def assignAllClusterIndices(self, medoids: torch.Tensor) -> None:
        # Chunking solution to memory problem: Compact3D
        offset: int = 10000
        i: int = 0
        while True:
            dist: torch.Tensor = torch.cdist(self._data[i * offset : (i + 1) * offset, : ], medoids, p=2) # L^2 norm
            self._indices[i * offset : (i + 1) * offset] = torch.argmin(dist, dim=1)

            i += 1
            if i * offset > self._data.shape[0]:
                break

    def findMedoids(self, clusters: list) -> torch.Tensor:
        ret: torch.Tensor = torch.empty(len(clusters), clusters[0].shape[1]).to(device=self._device)
        for i, cluster in enumerate(clusters):
            ret[i] = cluster[findMedoids(cluster)]
        return ret

    def initSampleSets(self, sampleSize: int) -> tuple:
        splits: tuple = self._data[np.random.choice(
                self._data.shape[0],
                sampleSize * self._numSampleSets,
                replace=True
        )].split(sampleSize)
        sampleSets = torch.stack(list(splits), dim=0).to(self._device) # Convert back to tensor

        samplesMaps: torch.tensor = torch.empty(self._numSampleSets, sampleSize, 2, dtype=torch.int32).to(self._device)
        samplesMaps[0].index_fill_(1, torch.tensor([0]).to(self._device), -1)
        samplesMaps[1:].index_fill_(2, torch.tensor([0, 1]).to(self._device), -2)
        for i in range(samplesMaps[0].shape[0]):
            samplesMaps[0, i, 1] = i
        
        # Because of only initializing sampleMaps[0] tail pointer, it requires an extra if test on the GPU -> faster than initializing here
        return sampleSets, samplesMaps    

    def run(self) -> tuple:
        with torch.no_grad():
            sampleSize: int = int(np.ceil(self._numClusters * np.log(self._numClusters) + self._GAMMA * self._numClusters))
            while True:
                sampleSets, sampleMaps = self.initSampleSets(sampleSize)
                for i in range(self._numSampleSets - 1):
                    mappingLength: int = euclidDistanceMapped(i, sampleSets[i], sampleSets[i + 1], sampleMaps[i], sampleMaps[i + 1], sampleMaps[0])
                    if mappingLength < self._numClusters:
                        sampleSize = sampleSize + int(np.ceil(sampleSize * (self._numSampleSets - i) / self._numSampleSets))
                        break
                diff: int = mappingLength - self._numClusters
                if diff > 0:
                    if diff <= self._MARGIN:
                        break
                    sampleSize = int(np.ceil(sampleSize * 0.95))
                    print("Difference", diff)

            medoids: torch.Tensor = self.findMedoids(collectGroups(sampleSize, self._numClusters, sampleSets, sampleMaps))
            self.assignAllClusterIndices(medoids)

        return medoids, self._indices

class Compressor:
    def __init__(
            self, device: torch.device,
            DCSHIterations: int,
            ScRoIterations: int,
            DCSHClusters: int,
            ScRoClusters: int,
            numSampleSets: int,
            batchSize: int = 100000
        ):
        self._device: torch.device = device
        self._batchSize: int = batchSize

        # Using cluster parameters as described in "Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector Quantization"
        self._DCSHIterations: int = DCSHIterations
        self._ScRoIterations: int = ScRoIterations

        self._DCSHClusters: int = DCSHClusters
        self._ScRoClusters: int = ScRoClusters

        self._numSampleSets: int = numSampleSets

    def compress(self, gaussians: GaussianModel, path: str, agoras: bool = False) -> None:
        with torch.no_grad():
            if not agoras:
                predictTotalTime(self._device, gaussians, self._DCSHIterations, self._DCSHClusters, self._ScRoIterations, self._ScRoClusters)
            
            t0 = time.time()
            save = dict()
            save["mean"] = gaussians._xyz.half().cpu()
            save["opacity"] = gaussians._opacity.half().cpu()

            save["cDCSH"], save["iDCSH"] = self.compressDCSH(gaussians.get_features_flat, agoras)
            save["cDCSH"] = save["cDCSH"].cpu(); save["iDCSH"] = save["iDCSH"].short().cpu()
            print("Done DCSH")

            save["cSc"], save["iSc"] = self.compressSc(gaussians._scaling, agoras)
            save["cSc"] = save["cSc"].cpu(); save["iSc"] = save["iSc"].short().cpu()
            print("Done scaling")

            save["cRo"], save["iRo"] = self.compressRo(gaussians._rotation, agoras)
            save["cRo"] = save["cRo"].cpu(); save["iRo"] = save["iRo"].short().cpu()
            print("Done rotation")

            np.savez_compressed(path, **save) # save with keyword names
            print("Finished in:", (time.time() - t0) / 60)

    def uncompress(self, gaussians, path: str) -> None:
        with torch.no_grad():
            gaussianPly: GaussianModel = GaussianModel(3)
            gaussianPly._xyz = torch.from_numpy(gaussians["mean"]).float()
            gaussianPly._opacity = torch.from_numpy(gaussians["opacity"]).float()
            gaussianPly._features_dc = torch.empty(gaussianPly._xyz.shape[0], 1, 3)
            gaussianPly._features_rest = torch.empty(gaussianPly._xyz.shape[0], 15, 3)
            gaussianPly._scaling = torch.empty(gaussianPly._xyz.shape[0], 3)
            gaussianPly._rotation = torch.empty(gaussianPly._xyz.shape[0], 4)

            cDCSH = torch.from_numpy(gaussians["cDCSH"]).float(); iDCSH = torch.from_numpy(gaussians["iDCSH"]).int()
            cDC = cDCSH[:, 0:3]
            cSH = cDCSH[:, 3:].reshape([cDCSH.shape[0], 15, 3])
            cSc = torch.from_numpy(gaussians["cSc"]).float(); iSc = torch.from_numpy(gaussians["iSc"]).int()
            cRo = torch.from_numpy(gaussians["cRo"]).float(); iRo = torch.from_numpy(gaussians["iRo"]).int()
            for i in range(gaussianPly._xyz.shape[0]):
                gaussianPly._features_dc[i, 0, :] = cDC[iDCSH[i]]
                gaussianPly._features_rest[i,] = cSH[iDCSH[i]]
                gaussianPly._scaling[i,] = cSc[iSc[i]]
                gaussianPly._rotation[i,] = cRo[iRo[i]]

            gaussianPly.save_ply(path)

    def compressDCSH(self, dcsh: torch.Tensor, agoras: bool = False) -> tuple:
        if agoras:
            agoras: AGORAS = AGORAS(dcsh, numSampleSets=5, numClusters=self._DCSHClusters, device=self._device)
            return agoras.run()
        kmeans: Kmeans = Kmeans(dcsh, self._device); kmeans.initClusters(numClusters=self._DCSHClusters)
        return kmeans.runIterations(numIterations=self._DCSHIterations, batchSize=self._batchSize)

    def compressSc(self, sc: torch.Tensor, agoras: bool = False) -> tuple:
        if agoras:
            agoras: AGORAS = AGORAS(sc, numClusters=self._ScRoClusters, numSampleSets=self._numSampleSets, device=self._device)
            return agoras.run()
        kmeans: Kmeans = Kmeans(sc, self._device); kmeans.initClusters(numClusters=self._ScRoClusters)
        return kmeans.runIterations(numIterations=self._ScRoIterations, batchSize=self._batchSize)

    def compressRo(self, ro: torch.Tensor, agoras: bool = False) -> tuple:
        if agoras:
            agoras: AGORAS = AGORAS(ro, numClusters=self._ScRoClusters, numSampleSets=self._numSampleSets, device=self._device)
            return agoras.run()
        kmeans: Kmeans = Kmeans(ro, self._device); kmeans.initClusters(numClusters=self._ScRoClusters)
        return kmeans.runIterations(numIterations=self._ScRoIterations, batchSize=self._batchSize)

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser("Compression parameters")
    parser.add_argument("-u", "--uncompress", action="store_true")
    parser.add_argument("-a", "--agoras", action="store_true")
    parser.add_argument("-cc", "--colorCl", default=2 ** 12, type=int)
    parser.add_argument("-gc", "--geomCl", default=2 ** 12, type=int)
    parser.add_argument("-ci", "--colorIt", default=100, type=int)
    parser.add_argument("-gi", "--geomIt", default=3000, type=int)
    parser.add_argument("-m", "--numSampl", default=50, type=int)
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    compressor: Compressor = Compressor(
        device,
        DCSHIterations=args.colorIt,
        ScRoIterations=args.geomIt,
        DCSHClusters=args.colorCl,
        ScRoClusters=args.geomCl,
        numSampleSets=args.numSampl
    )
    if args.uncompress:
        compressor.uncompress(np.load(args.input), args.output)
    else:
        gaussians = GaussianModel(3)
        gaussians.load_ply(args.input)
        compressor.compress(gaussians, args.output, args.agoras)

# Means, opacity - f16 (also on GPU, same for cov)
# indices - i16
# Codebooks - f32 (to easily work with on the CPU and less unpacking on the GPU) (codebooks are only < %5 of memory)