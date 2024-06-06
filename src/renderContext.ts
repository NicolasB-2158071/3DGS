export class RenderContext {
    public canvas: HTMLCanvasElement;
    public W: number;
    public H: number;
    
    public device: GPUDevice;
    public context: GPUCanvasContext;
    public settings: any;

    private canvasToSizeMap: WeakMap<any, any>;
    private resizeObserver: ResizeObserver;

    constructor(canvas: HTMLCanvasElement, device: GPUDevice, settings: any) {
        this.canvas = canvas;
        this.W = this.canvas.width;
        this.H = this.canvas.height;
        this.settings = settings;

        this.device = device;
        this.context = this.canvas.getContext("webgpu");
        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: "premultiplied"
        });

        this.canvasToSizeMap = new WeakMap();
        this.resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                this.canvasToSizeMap.set(entry.target, {
                    width: entry.contentBoxSize[0].inlineSize,
                    height: entry.contentBoxSize[0].blockSize,
                });
            }
        });
        this.resizeObserver.observe(this.canvas);
    }

    destroy() { 
        this.device.destroy();
        this.device = null as any;
    }

    public resizeCanvasToDisplaySize() {
        // Get the canvas's current display size
        let { width, height } = this.canvasToSizeMap.get(this.canvas) || this.canvas;
    
        // Make sure it's valid for WebGPU
        this.W = Math.max(1, Math.min(width, this.device.limits.maxTextureDimension2D));
        this.H = Math.max(1, Math.min(height, this.device.limits.maxTextureDimension2D));
    
        // Only if the size is different, set the canvas size
        const needResize = this.canvas.width !== width || this.canvas.height !== height;
        if (needResize) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
        return needResize;
    }
    
}