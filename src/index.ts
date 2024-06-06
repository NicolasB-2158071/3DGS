import { RenderContext } from "./renderContext";
import { Renderer } from "./renderer";
import { initGUI } from "./gui";

// Less painfull to have this one globally
async function initWebGPU(): Promise<GPUDevice> {
    if (!navigator.gpu) {
        alert("WebGPU not supported!");
        return;
    }

    const adapter: GPUAdapter = await navigator.gpu.requestAdapter();
    const device: GPUDevice = await adapter.requestDevice({
        requiredLimits: {
            maxComputeWorkgroupStorageSize: 17500,
        },
        requiredFeatures: ["timestamp-query"]
    });
    if (!adapter || !device) {
        alert("WebGPU not supported!");
        return;
    }
    return device;
}

async function main(): Promise<void> {
    let settings: any = {
        uploadDataFile: () => document.getElementById("data").click(),
        uploadCameraJson: () => document.getElementById("camera").click(),
        viewPoints: "",
        scalingModifier: 1,
        cameraSpeed: 0.05,
        backgroundColor: "#000000",
        fps: 0,
        preprocessTime: 0,
        sortTime: 0,
        renderTime: 0
    };
    initGUI(settings);
    
    const renderer: Renderer = new Renderer(new RenderContext(document.getElementById("c") as HTMLCanvasElement, await initWebGPU(), settings));
    let then: number = 0;
    function loop(now: number) {
        document.body.style.backgroundColor = settings.backgroundColor;

        now *= 0.001;
        const deltaTime: number = now - then;
        then = now;
        renderer.frame();

        settings.fps = (1.0 / deltaTime).toFixed(1);
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}

main();