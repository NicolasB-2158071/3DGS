import { PlyLoader, LoadedGaussiansPly } from "../loaders/plyLoader";
import { Float16Array } from "@petamoriken/float16";
import { NpzLoader, LoadedGaussiansNpz } from "../loaders/npzLoader";
import { Camera, CameraViewPoint } from "./camera";
import { mat3 } from "wgpu-matrix";
import { gui } from "../gui";

export class Scene {
    public camera: Camera;
    private viewPoints: Array<CameraViewPoint>;

    private rawGaussians: GPUBuffer;
    private processedGaussians: GPUBuffer;
    private shCoefficients: GPUBuffer;

    private compressed: boolean;
    private numGaussians: number;
    private sceneLoaded: boolean;

    constructor(width: number, height: number) {
        this.camera = new Camera(width, height);

        this.numGaussians = 0;
        this.sceneLoaded = false;
    }

    public async setScene(scene: File, device: GPUDevice, finished: Function): Promise<void> {
        let ext: string = scene.name.slice(scene.name.lastIndexOf("."));
        if (ext === ".npz") {
            this.compressed = true;
            let npzLoader: NpzLoader = new NpzLoader();
            npzLoader.loadNpz(scene, () => {
                this.numGaussians = npzLoader.getNumGaussians();
                this.initGaussianBuffersNpz(device, npzLoader.getLoadedGaussians());
                finished();
            });
        }
        else {
            this.compressed = false;
            let plyLoader: PlyLoader = new PlyLoader();
            await plyLoader.loadPly(scene);
            this.numGaussians = plyLoader.getNumGaussians();
            this.initGaussianBuffersPly(device, plyLoader.getLoadedGaussians());
            finished();
        }
    }

    private initGaussianBuffersPly(device: GPUDevice, loadedGaussians: LoadedGaussiansPly): void {
        this.rawGaussians = device.createBuffer({
            label: "Raw gaussians buffer",
            mappedAtCreation: true,
            size: this.numGaussians * 64, // Offset of 4 bytes by mean (see website that calculates it)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const gaussiansValues = new Float32Array(this.rawGaussians.getMappedRange());
        for (let i = 0; i < this.numGaussians; ++i) {
            const offset: number = i * 16;
            gaussiansValues.set(loadedGaussians.means.slice(i * 3, i * 3 + 3), 0 + offset);
            gaussiansValues.set(loadedGaussians.colors.slice(i * 3, i * 3 + 3), 4 + offset); // padding
            gaussiansValues[7 + offset] = loadedGaussians.opacities[i];
            gaussiansValues.set(loadedGaussians.cov3Ds.slice(i * 6, i * 6 + 6), 8 + offset);
        }
        this.rawGaussians.unmap(); // https://toji.dev/webgpu-best-practices/buffer-uploads.html

        this.processedGaussians = device.createBuffer({
            label: "Processed gaussians buffer",
            size: this.numGaussians * 20,
            usage: GPUBufferUsage.STORAGE
        });
    }

    private initGaussianBuffersNpz(device: GPUDevice, loadedGaussians: LoadedGaussiansNpz): void {
        this.rawGaussians = device.createBuffer({
            label: "Raw gaussians buffer",
            mappedAtCreation: true,
            size: this.numGaussians * 24,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const gaussiansValues = this.rawGaussians.getMappedRange();
        for (let i = 0; i < this.numGaussians; ++i) {
            const offset: number = i * 24;
            const views = {
                mean: new Uint32Array(gaussiansValues, 0 + offset, 2),
                indexSH: new Uint32Array(gaussiansValues, 8 + offset, 1),
                cov3D: new Uint32Array(gaussiansValues, 12 + offset, 3)
            };
            let meanView: Float16Array = new Float16Array(views.mean.buffer, 0 + offset, 4);
            meanView.set(loadedGaussians.means.slice(i * 3, (i + 1) * 3), 0); meanView[3] = loadedGaussians.opacities[i]; 
            views.indexSH[0] = loadedGaussians.indicesSH[i];

            let covView: Float16Array = new Float16Array(views.cov3D.buffer, 12 + offset, 6);
            covView.set(loadedGaussians.cov3Ds.slice(i * 6, (i + 1) * 6), 0);
        }
        this.rawGaussians.unmap(); // https://toji.dev/webgpu-best-practices/buffer-uploads.html

        this.processedGaussians = device.createBuffer({
            label: "Processed gaussians buffer",
            size: this.numGaussians * 20,
            usage: GPUBufferUsage.STORAGE
        });

        let numberClusters: number = loadedGaussians.SH.length / 48;
        this.shCoefficients = device.createBuffer({
            label: "SH coefficients buffer",
            mappedAtCreation: true,
            size: loadedGaussians.SH.byteLength + 64 * numberClusters,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const shValues = new Float32Array(this.shCoefficients.getMappedRange());
        for (let i = 0; i < numberClusters; ++i) {
            for (let j = 0; j < 16; ++j) {
                shValues.set(
                    loadedGaussians.SH.slice(
                    3 * j + 48 * i, 3 * (j + 1) + 48 * i),
                    4 * j + 64 * i); // 1 padding
            }
        }
        this.shCoefficients.unmap();
    }

    public setCameraViewPoints(viewPoints: File, settings: any): void {
        const reader: FileReader = new FileReader();
        reader.onload = (evt) => {
            this.viewPoints = Array<CameraViewPoint>();
            const viewPoints: {[key: number]: any} = JSON.parse(evt.target.result as string);
            for (const key in viewPoints) {
                this.viewPoints.push({
                    name: viewPoints[key].img_name,
                    rotation: mat3.create(...viewPoints[key].rotation.flat()),
                    position: viewPoints[key].position
                } as CameraViewPoint)
            }
            let entries: {[key: string]: number} = {};
            this.viewPoints.forEach((entry: CameraViewPoint, index: number) => {
                entries[entry.name] = index;
            });
            gui.folders[1].add(settings, "viewPoints", entries).name("view points").onChange(() => {
                this.camera.setViewPoint(this.viewPoints[settings.viewPoints]);
            });
        };
        reader.onerror = (evt) => {
            console.log("Failed to read view points", evt);
        }
        reader.readAsText(viewPoints);
    }

    public setSceneLoaded(loaded: boolean): void {
        this.sceneLoaded = loaded;
    }

    public isSceneLoaded(): boolean {
        return this.sceneLoaded;
    }

    public isCompressed(): boolean {
        return this.compressed;
    }

    public getNumGaussians(): number {
        return this.numGaussians;
    }

    public getRawGaussianBuffer(): GPUBuffer {
        return this.rawGaussians;
    }

    public getProcessedGaussianBuffer(): GPUBuffer {
        return this.processedGaussians;
    }

    public getShCoefficientsBuffer(): GPUBuffer {
        return this.shCoefficients;
    }
}