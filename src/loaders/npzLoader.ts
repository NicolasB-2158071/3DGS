import * as JSZip from "jszip";
import npyjs from "npyjs";
import { sigmoid, normalizeQuaternion, expArray, computeCov3D } from "../utils/maths";
import { Float16Array } from "@petamoriken/float16";

export interface LoadedGaussiansNpz {
    means: Float16Array,
    opacities: Float16Array,
    SH: Float32Array, // Harder to pack this on the GPU
    indicesSH: Uint32Array, // Read as Uint32Array instead of Int16Array
    cov3Ds: Float16Array
}

export class NpzLoader {
    private means: Float16Array;
    private opacities: Float16Array;
    private cov3Ds: Float16Array;
    private SH: Float32Array;
    private indicesSH: Uint32Array;

    private numGaussians: number;

    // SLOW - Float16Array is not native, Proxy implementations has significant extra overhead
    public loadNpz(npzFile: File, finished: Function) {
        const start: number = performance.now(); // DEBUG

        let fileReader: FileReader = new FileReader();
        fileReader.onload = () => {
            let zipper: JSZip = JSZip(); // loadAsync merges with zipper
            let n: npyjs = new npyjs();
            zipper.loadAsync(fileReader.result).then(async (zip: JSZip) => {
                this.means = (await n.load(await zip.files["mean.npy"].async("arraybuffer"))).data as Float16Array;
                this.opacities = (await n.load(await zip.files["opacity.npy"].async("arraybuffer"))).data as Float16Array;
                this.opacities.forEach((value: number, index: number, array: Float16Array) => {array[index] = sigmoid(value);});

                this.indicesSH = new Uint32Array((await n.load(await zip.files["iDCSH.npy"].async("arraybuffer"))).data);
                this.numGaussians = this.indicesSH.length;
                this.initCovarianceMatrices(
                    (await n.load(await zip.files["cSc.npy"].async("arraybuffer"))).data as Float32Array,
                    (await n.load(await zip.files["iSc.npy"].async("arraybuffer"))).data as Int16Array,
                    (await n.load(await zip.files["cRo.npy"].async("arraybuffer"))).data as Float32Array,
                    (await n.load(await zip.files["iRo.npy"].async("arraybuffer"))).data as Int16Array
                );
                this.SH = (await n.load(await zip.files["cDCSH.npy"].async("arraybuffer"))).data as Float32Array;

                console.log(`Loaded ${this.numGaussians} gaussians in ${((performance.now() - start) / 1000).toFixed(3)}s`); // DEBUG
                finished();
            });
        }
        fileReader.onerror = (err) => {
            console.log("Failed to read npz file");
            console.log(err);
        }
        fileReader.readAsArrayBuffer(npzFile);
    }

    private initCovarianceMatrices(
        codeBookScale: Float32Array,
        indicesScale: Int16Array,
        codebookRotation: Float32Array,
        indicesRotation: Int16Array
    ): void {
        this.cov3Ds = new Float16Array(this.numGaussians * 6);
        for (let i = 0; i < this.numGaussians; ++i) {
            const scaleClusterId: number = indicesScale[i];
            const rotationClusterId: number = indicesRotation[i];

            this.cov3Ds.set(computeCov3D(
                expArray(codeBookScale.slice(scaleClusterId * 3, (scaleClusterId + 1) * 3)),
                normalizeQuaternion(codebookRotation.slice(rotationClusterId * 4, (rotationClusterId + 1) * 4))
            ), i * 6);
        }
    }

    public getNumGaussians(): number {
        return this.numGaussians;
    }

    public getLoadedGaussians(): LoadedGaussiansNpz {
        return {
            means: this.means,
            opacities: this.opacities,
            SH: this.SH,
            indicesSH: this.indicesSH,
            cov3Ds: this.cov3Ds
        };
    }
}

// cov berekenen, juiste buffers, sh_evaluation
// Implementeren: trainen, dispatched rendering, camera loaden
// Afweging clusters op GPU en telkens cov berekenen of meer data op GPU -> cov clusteren lost dit op, geeft goede resultaten?