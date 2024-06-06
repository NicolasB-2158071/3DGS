import { RenderContext } from "./renderContext";
import preprocessShader from "./shaders/preprocess.wgsl?raw"
import preprocessShaderComp from "./shaders/preprocessComp.wgsl?raw"
import renderShader from "./shaders/render.wgsl?raw";
import { GPUSorter, SortBuffers, guessWorkGroupSize } from "./radixSort";
import { Scene } from "./scene/scene";
import { PipelineBuilder } from "./utils/pipelineBuilder";
import { Timer } from "./utils/timer";

export class Renderer {
    private timer: Timer;
    private scene: Scene;

    private renderContext: RenderContext;
    private radixSorter: GPUSorter;
    private sortBuffers: SortBuffers;

    private pipelineBuilder: PipelineBuilder;
    private renderPipeline: GPURenderPipeline;
    private preprocessPipeline: GPUComputePipeline;

    private preprocessBindGroup: GPUBindGroup;
    private sorterBindGroup: GPUBindGroup;
    private renderBindGroup: GPUBindGroup;

    private sortIndirectBuffer: GPUBuffer;
    private renderIndirectBuffer: GPUBuffer;

    private passDescriptor: GPURenderPassDescriptor;
    private uniforms: GPUBuffer;

    constructor(renderContext: RenderContext) {
        this.renderContext = renderContext;
        this.timer = new Timer(this.renderContext.device);
        this.scene = new Scene(this.renderContext.W, this.renderContext.H);

        this.pipelineBuilder = new PipelineBuilder();

        this.initIndirectBuffers();
        this.initUniforms();
        this.initPassDescriptor();

        guessWorkGroupSize(this.renderContext.device, this.renderContext.device.queue).then((value: number) => {
            this.radixSorter = new GPUSorter(this.renderContext.device, value, this.timer);
        });
        
        const inputFields: HTMLCollectionOf<HTMLInputElement> = document.getElementsByTagName("input") as HTMLCollectionOf<HTMLInputElement>;
        inputFields[0].addEventListener("change", async () => {
            this.scene.setScene(inputFields[0].files[0], this.renderContext.device, async () => {
                this.sortBuffers = this.radixSorter.create_sort_buffers(this.renderContext.device, this.scene.getNumGaussians());
                this.initPipelines()
                this.initBindGroups();
                this.scene.setSceneLoaded(true); // Cheasy but async asked for it..
            });
        });
        inputFields[1].addEventListener("change", () => {
            this.scene.setCameraViewPoints(inputFields[1].files[0], this.renderContext.settings);
        });
    }

    private initPipelines(): void {
        let modules: [GPUShaderModule, GPUShaderModule] = this.initShaderModules();
        this.preprocessPipeline = this.pipelineBuilder.buildPreprocessPipeline(
            this.renderContext.device,
            modules[1],
            this.pipelineBuilder.buildPreprocessPipelineLayout(
                this.renderContext.device,
                [
                    this.pipelineBuilder.buildPreprocessBindGroupLayout(this.renderContext.device, this.scene.isCompressed()),
                    this.pipelineBuilder.buildSorterBindGroupLayout(this.renderContext.device)
                ]
            )
        );
        this.renderPipeline = this.pipelineBuilder.buildRenderPipeline(
            this.renderContext.device,
            modules[0],
            this.pipelineBuilder.buildRenderPipelineLayout(
                this.renderContext.device,
                [
                    this.pipelineBuilder.buildRenderBindGroupLayout(this.renderContext.device)
                ]
            )
        );
    }

    private initShaderModules(): [GPUShaderModule, GPUShaderModule] {
        let renderShaderModule: GPUShaderModule = this.renderContext.device.createShaderModule({
            label: "Main render shader module",
            code: renderShader
        });

        let preprocessShaderModule: GPUShaderModule = this.renderContext.device.createShaderModule({
            label: "Preprocess shader module",
            code: this.scene.isCompressed() ? preprocessShaderComp : preprocessShader
        });

        return [renderShaderModule, preprocessShaderModule] 
    }

    private initIndirectBuffers(): void {
        this.sortIndirectBuffer = this.renderContext.device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE
        });
        const sortIndirectData: Uint32Array = new Uint32Array(3);
        sortIndirectData[0] = 0; // Set this in preprocess shader (workgroupX)
        sortIndirectData[1] = 1;
        sortIndirectData[2] = 1;
        this.renderContext.device.queue.writeBuffer(this.sortIndirectBuffer, 0, sortIndirectData);
        
        this.renderIndirectBuffer = this.renderContext.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
        });
        const renderIndirectData: Uint32Array = new Uint32Array(4);
        renderIndirectData[0] = 4;
        renderIndirectData[1] = 0; // InstanceCount -> copy sort info over
        renderIndirectData[2] = 0;
        renderIndirectData[3] = 0;
        this.renderContext.device.queue.writeBuffer(this.renderIndirectBuffer, 0, renderIndirectData);
    }

    private initUniforms(): void {
        this.uniforms = this.renderContext.device.createBuffer({
            size: 7 * 4 + 3 * 4 + 32 * 4 + 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    private writeUniforms(): void {
        const uniformValues: Float32Array = new Float32Array(44);
        uniformValues[0] = this.renderContext.W;
        uniformValues[1] = this.renderContext.H;
        uniformValues[2] = this.scene.camera.focalX;
        uniformValues[3] = this.scene.camera.focalY;
        uniformValues[4] = this.scene.camera.tanFovX;
        uniformValues[5] = this.scene.camera.tanFovY;
        uniformValues[6] = this.renderContext.settings.scalingModifier;
        uniformValues.set(this.scene.camera.getPosition(), 8);
        uniformValues.set(this.scene.camera.getViewMatrix(), 12);
        uniformValues.set(this.scene.camera.getVP(), 28);

        this.renderContext.device.queue.writeBuffer(
            this.uniforms,
            0,
            uniformValues
        );
    }

    private initPassDescriptor(): void {
        this.timer.subscribe("renderTime");
        this.timer.subscribe("preprocessTime");
        this.passDescriptor = {
            colorAttachments: [
                {
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                    loadOp: "clear",
                    storeOp: "store"
                } as GPURenderPassColorAttachment
            ],
            timestampWrites: this.timer.getSubscription("renderTime")
        }
    }

    private initBindGroups(): void {
        let descriptor: GPUBindGroupDescriptor = {
            label: "BindGroup used for preprocessing",
            layout: this.pipelineBuilder.buildPreprocessBindGroupLayout(this.renderContext.device, this.scene.isCompressed()),
            entries: [
              { binding: 0, resource: { buffer: this.uniforms }},
              { binding: 1, resource: { buffer: this.scene.getRawGaussianBuffer() }},
              { binding: 2, resource: { buffer: this.scene.getProcessedGaussianBuffer() }},
            ]
        }
        if (this.scene.isCompressed()) {
            (descriptor.entries as Array<GPUBindGroupEntry>).push({ binding: 3, resource: { buffer: this.scene.getShCoefficientsBuffer() }})
        }
        this.preprocessBindGroup = this.renderContext.device.createBindGroup(descriptor);
        this.sorterBindGroup = this.renderContext.device.createBindGroup({
            label: "BindGroup used for sorting",
            layout: this.pipelineBuilder.buildSorterBindGroupLayout(this.renderContext.device),
            entries: [
                { binding: 0, resource: { buffer: this.sortBuffers.keys() }}, // depths
                { binding: 1, resource: { buffer: this.sortBuffers.values() }}, // indices
                { binding: 2, resource: { buffer: this.sortIndirectBuffer }},
                { binding: 3, resource: { buffer: this.sortBuffers.state_buffer() }}
            ]
        });
        this.renderBindGroup = this.renderContext.device.createBindGroup({
            label: "BindGroup used for rendering",
            layout: this.pipelineBuilder.buildRenderBindGroupLayout(this.renderContext.device),
            entries: [
              { binding: 0, resource: { buffer: this.uniforms }},
              { binding: 1, resource: { buffer: this.scene.getProcessedGaussianBuffer() }},
              { binding: 2, resource: { buffer: this.sortBuffers.values() }}
            ]
        });
    }

    private resetSortBuffers(): void {
        this.renderContext.device.queue.writeBuffer(this.sortIndirectBuffer, 0, new Uint32Array([0])); // Reset workgroupX
        this.renderContext.device.queue.writeBuffer(this.sortBuffers.state_buffer(), 0, new Uint32Array([0])); // Reset num_keys
    }

    private updateTimings(): void {
        this.timer.resolveResults();
        this.renderContext.settings.preprocessTime = this.timer.getResultByName("preprocessTime");
        this.renderContext.settings.sortTime = this.timer.getResultByIds(this.timer.getIdOfName("sortScatterTime"), this.timer.getIdOfName("sortHistogramTime"));
        this.renderContext.settings.renderTime = this.timer.getResultByName("renderTime");
    }

    private preprocess(commandEncoder: GPUCommandEncoder): void {
        const passEncoder: GPUComputePassEncoder = commandEncoder.beginComputePass({label: "Compute pass", timestampWrites: this.timer.getSubscription("preprocessTime")});
        passEncoder.setPipeline(this.preprocessPipeline);
        passEncoder.setBindGroup(0, this.preprocessBindGroup);
        passEncoder.setBindGroup(1, this.sorterBindGroup);

        let workgroupAmount: number = Math.ceil(this.scene.getNumGaussians() / 256);
        passEncoder.dispatchWorkgroups(workgroupAmount, 1, 1);
        passEncoder.end();
    }

    private render(commandEncoder: GPUCommandEncoder): void {
        commandEncoder.copyBufferToBuffer(
            this.sortBuffers.state_buffer(),
            0,
            this.renderIndirectBuffer,
            4, // InstanceCount
            4 // Copy num_keys
        );

        const passEncoder: GPURenderPassEncoder = commandEncoder.beginRenderPass(this.passDescriptor);
        passEncoder.setPipeline(this.renderPipeline);
        passEncoder.setBindGroup(0, this.renderBindGroup);

        passEncoder.drawIndirect(this.renderIndirectBuffer, 0);
        passEncoder.end();
    }

    public frame(): void {
        if (!this.scene.isSceneLoaded()) {return;}
        if (this.renderContext.resizeCanvasToDisplaySize()) {
            this.scene.camera.initProjectionMatrix(this.renderContext.W, this.renderContext.H);
        }
        if (!this.scene.camera.update(this.renderContext.settings.cameraSpeed)) {return;}
        
        (this.passDescriptor.colorAttachments as Array<GPURenderPassColorAttachment>)[0].view = this.renderContext.context.getCurrentTexture().createView(); // TS??!
        const commandEncoder = this.renderContext.device.createCommandEncoder();
        
        this.writeUniforms();
        this.resetSortBuffers();

        this.preprocess(commandEncoder);
        this.radixSorter.sort_indirect(commandEncoder, this.sortBuffers, this.sortIndirectBuffer, this.timer);
        this.render(commandEncoder);

        this.timer.submitTimestamps(commandEncoder);
        this.renderContext.device.queue.submit([commandEncoder.finish()]);
        this.updateTimings();
    }
}