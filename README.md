# 3D Gaussian splatting

WebGPU renderer and VQ compression

## Renderer
To install (Webpack, TS, ..)

```
npm install
```

To build (dist folder) use one of the following

```
npm run build
npm run watch
```

*Makes use of a TS conversion of [KeKsBoTer's wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort)*

## Compression

No conda 🙃, just keep pipping untill all dependencies are found.

To compile the custom CUDA kernels, run

```
python setup.py install
```

*Note: the PyTorch version needs to match the installed CUDA version*