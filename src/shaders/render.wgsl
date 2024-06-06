struct Uniforms {
    W: f32,
    H: f32,
    focalX: f32,
    focalY: f32,
    tanFovX: f32,
    tanFovY: f32,
    scalingModifier: f32,
    viewMatrix: mat4x4<f32>,
    viewProjectionMatrix: mat4x4<f32>
}

struct ProcessedGaussian {
    colorRG: u32, colorBA: u32, // avoiding vec
    conicXY: u32, conicZR: u32,
    pointImage: u32
}

struct GaussianOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) con_o: vec4<f32>,
    @location(2) xy: vec2<f32>,
    @location(3) pixf: vec2<f32>,
    @location(4) scalingModifier: f32
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> processedGaussians: array<ProcessedGaussian>;
@group(0) @binding(2) var<storage, read> valuesIndices: array<u32>;

const quadVertices = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0)
);

@vertex
fn vertexShader(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> GaussianOutput {
    let processedGaussian: ProcessedGaussian = processedGaussians[valuesIndices[instanceIndex]];
    let color: vec4<f32> = vec4<f32>(unpack2x16float(processedGaussian.colorRG), unpack2x16float(processedGaussian.colorBA));
    var conicR: vec4<f32> = vec4<f32>(unpack2x16float(processedGaussian.conicXY), unpack2x16float(processedGaussian.conicZR));
    let pointImage: vec2<f32> = unpack2x16float(processedGaussian.pointImage);

    conicR.w *= 0.15 + uniforms.scalingModifier * 0.85;
    let screenPos: vec2<f32> = pointImage + conicR.w * quadVertices[vertexIndex];
    let clipPos: vec2<f32> = screenPos / vec2<f32>(uniforms.W, uniforms.H) * 2.0 - 1.0;

    var output: GaussianOutput;
    output.scalingModifier = 1.0 / uniforms.scalingModifier;
    output.color = color.rgb;
    output.position = vec4<f32>(clipPos, 0.0, 1.0);
    output.con_o = vec4<f32>(conicR.xyz, color.a);
    output.xy = pointImage; // Center gaussian (pixel)
    output.pixf = screenPos; // Corner box (pixel)

    return output;
}

@fragment
fn fragmentShader(input: GaussianOutput) -> @location(0) vec4<f32> {
    // Resample using conic matrix (cf. "Surface 
    // Splatting" by Zwicker et al., 2001)
    let d: vec2<f32> = input.xy - input.pixf; // Coordinate relative to ellipse (distance x, y from mean)
    var power: f32 = -0.5 * (input.con_o.x * d.x * d.x + input.con_o.z * d.y * d.y) - input.con_o.y * d.x * d.y; // Formula G(x)
    // var power: f32 = -0.5 * (input.con_o.x * d.x * d.x + input.con_o.z * d.y * d.y + 2 * input.con_o.y * d.x * d.y); // See notes

    if (power > 0.0) {
        discard;
    }
    power *= input.scalingModifier;

    // Eq. (2) from 3D Gaussian splatting paper.
    let alpha: f32 = min(0.99, input.con_o.w * exp(power)); // Opacity exponential decay from mean, min to avoid numerical instabilities
    if (alpha < 1.0 / 255.0) {
        discard;
    }

    // Eq. (3) from 3D Gaussian splatting paper.
    return vec4<f32>(input.color * alpha, alpha);
}