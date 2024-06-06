import { sigmoid, normalizeQuaternion, expArray, computeCov3D } from "../utils/maths";

interface Gaussian {
    mean: Float32Array,
    harmonic: Float32Array,
    opacity: number,
    scale: Float32Array,
    rotation: Float32Array
}

export interface LoadedGaussiansPly {
    means: Array<number>,
    colors: Array<number>,
    opacities: Array<number>,
    cov3Ds: Array<number>
}

export class PlyLoader {
    // Represent it as a sequence of numbers for minimization
    private means: Array<number>;
    private colors: Array<number>;
    private opacities: Array<number>;
    private cov3Ds: Array<number>;

    private numGaussians: number;
    private vertexStartOffset: number;

    private PLY_HEADER_SIZE: number = 2000; // First couple bytes are ascii header
    private FLOATS_PER_GAUSSIAN: number = 62;
    private SH_C0: number = 0.28209479177387814;

    constructor() {
        this.means = [];
        this.colors = [];
        this.opacities = [];
        this.cov3Ds = [];
    }

    public async loadPly(plyFile: File) {
        const start: number = performance.now(); // DEBUG

        let fileData: ArrayBuffer = await this.readPlyFileAsArrayBuffer(plyFile);
        this.decodeHeader(fileData);

        const dataView: DataView = new DataView(fileData, this.vertexStartOffset); // Deals with endianness
        for (let i: number = 0; i < this.numGaussians; ++i) {
            let gaussian: Gaussian = this.extractGaussian(dataView, i);

            this.means.push(...(gaussian.mean));
            const cov3D: Array<number> = computeCov3D(expArray(gaussian.scale), normalizeQuaternion(gaussian.rotation)); // Precompute Cov3D (because only rendering)
            this.cov3Ds.push(...cov3D);

            // TEMP: precomputed SH (degree 0)
            const color: Array<number> = [
                0.5 + this.SH_C0 * gaussian.harmonic[0],
                0.5 + this.SH_C0 * gaussian.harmonic[1],
                0.5 + this.SH_C0 * gaussian.harmonic[2]
            ];
            this.colors.push(...color);
            this.opacities.push(sigmoid(gaussian.opacity)); // Opacity between 0 - 1
        }
        console.log(`Loaded ${this.numGaussians} gaussians in ${((performance.now() - start) / 1000).toFixed(3)}s`); // DEBUG
    }

    private readPlyFileAsArrayBuffer(plyFile: File): Promise<ArrayBuffer> { // Convert file to binary data
        return new Promise((resolve, reject) => {
            const reader: FileReader = new FileReader();
            reader.readAsArrayBuffer(plyFile);

            reader.onload = (event) => {
                if (typeof event.target.result === "string") {
                    console.log("Not a ply file");
                    return;
                }
                resolve(event.target.result);
            };
            reader.onerror = (event) => {
                reject(event.target.error);
            };
        });
    }

    private decodeHeader(fileData: ArrayBuffer): void {
        const header: string = new TextDecoder("utf-8").decode(fileData.slice(0, this.PLY_HEADER_SIZE)); // First 
        this.numGaussians = parseInt(header.match(/element vertex (\d+)/)[1]);
        this.vertexStartOffset = header.indexOf("end_header") + "end_header".length + 1;
    }

    private extractGaussian(dataView: DataView, gaussianId: number): Gaussian {
        let startOffset: number = gaussianId * this.FLOATS_PER_GAUSSIAN * 4;
        return {
            mean: (new Float32Array(3)).map((_: any, i: number) => dataView.getFloat32(startOffset + i * 4, true)),
            harmonic: (new Float32Array(3)).map((_: any, i: number) => dataView.getFloat32((startOffset + 6 * 4) + i * 4, true)),
            opacity: dataView.getFloat32(startOffset + 54 * 4, true),
            scale: (new Float32Array(3)).map((_: any, i: number) => dataView.getFloat32((startOffset + 55 * 4) + i * 4, true)), // 6 + 48 to the last harmonic
            rotation: (new Float32Array(4)).map((_: any, i: number) => dataView.getFloat32((startOffset + 58 * 4) + i * 4, true))
        };
    }

    public getNumGaussians(): number {
        return this.numGaussians;
    }

    public getLoadedGaussians(): LoadedGaussiansPly {
        return {
            means: this.means,
            colors: this.colors,
            opacities: this.opacities,
            cov3Ds: this.cov3Ds
        };
    }
}

// Based upon
// https://github.com/kishimisu/Gaussian-Splatting-WebGL/blob/main/src/loader.js