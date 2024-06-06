export class PipelineBuilder {
    public buildPreprocessBindGroupLayout(device: GPUDevice, compressed: boolean): GPUBindGroupLayout {
      let descriptor: GPUBindGroupLayoutDescriptor = {
        label: "BindGroupLayout used for preprocessing pipeline",
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" }
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" }
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" }
          }
        ]
      };
      if (compressed) {
        (descriptor.entries as Array<GPUBindGroupLayoutEntry>).push({binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" }})
      }
      
      return device.createBindGroupLayout(descriptor);
    }

    // Not used in the radixSort!! For preprocessing usage
    public buildSorterBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
      return device.createBindGroupLayout({
        label: "BindGroupLayout used for preprocessing pipeline",
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" }
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" }
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" }
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" }
          }
        ]
    });
    }

    public buildRenderBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
      return device.createBindGroupLayout({
          label: "BindGroupLayout used for rendering pipeline",
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.VERTEX,
              buffer: { type: "uniform" }
            },
            {
              binding: 1,
              visibility: GPUShaderStage.VERTEX,
              buffer: { type: "read-only-storage" }
            },
            {
              binding: 2,
              visibility: GPUShaderStage.VERTEX,
              buffer: { type: "read-only-storage" }
            }
          ]
      });
    }

    public buildPreprocessPipelineLayout(device: GPUDevice, groupLayouts: Iterable<GPUBindGroupLayout>): GPUPipelineLayout {
      return device.createPipelineLayout({
          bindGroupLayouts: groupLayouts // i layout is group(i)
      } as GPUPipelineLayoutDescriptor);
    }

    public buildRenderPipelineLayout(device: GPUDevice, groupLayouts: Iterable<GPUBindGroupLayout>): GPUPipelineLayout {
        return device.createPipelineLayout({
            bindGroupLayouts: groupLayouts // i layout is group(i)
        } as GPUPipelineLayoutDescriptor);
    }

    public buildPreprocessPipeline(device: GPUDevice, shaderModule: GPUShaderModule, pipelineLayout: GPUPipelineLayout): GPUComputePipeline {
      return device.createComputePipeline({
          layout: pipelineLayout,
          compute: {
              module: shaderModule,
              entryPoint: "preprocess"
          }
      } as GPUComputePipelineDescriptor);
    }

    public buildRenderPipeline(device: GPUDevice, shaderModule: GPUShaderModule, pipelineLayout: GPUPipelineLayout): GPURenderPipeline {
        return device.createRenderPipeline({
            vertex: {
              module: shaderModule,
              entryPoint: "vertexShader"
            },
            fragment: {
              module: shaderModule,
              entryPoint: "fragmentShader",
              targets: [
                {
                  format: navigator.gpu.getPreferredCanvasFormat(),
                  blend: {
                        color:
                        {
                            srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                            dstFactor: "one" as GPUBlendFactor,
                            operation: "add" as GPUBlendOperation
                        },
                        alpha:
                        {
                            srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                            dstFactor: "one" as GPUBlendFactor,
                            operation: "add" as GPUBlendOperation
                        }
                    }
                }
              ]
            },
            layout: pipelineLayout,
            primitive: {
                topology: "triangle-strip"
            }
        });
    }
}

// Seperate pipelines: vertex shaders can only have read-only storage buffers