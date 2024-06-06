struct Uniforms {
    W: f32,
    H: f32,
    focalX: f32,
    focalY: f32,
    tanFovX: f32,
    tanFovY: f32,
    scalingModifier: f32,
    pos: vec3<f32>,
    viewMatrix: mat4x4<f32>,
    viewProjectionMatrix: mat4x4<f32>
}

struct GeneralInfo {
    num_keys: atomic<u32>,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32
}

struct SortIndirect {
    dispatchX: atomic<u32>,
    dispatchY: u32,
    dispatchZ: u32
}

struct RawGaussian {
    mean: vec3<f32>,
    color: vec3<f32>,
    opacity: f32,
    cov3D: array<f32, 6>
}

struct ProcessedGaussian {
    colorRG: u32, colorBA: u32, // avoiding vec
    conicXY: u32, conicZR: u32,
    pointImage: u32
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> rawGaussians: array<RawGaussian>;
@group(0) @binding(2) var<storage, read_write> processedGaussians: array<ProcessedGaussian>;

@group(1) @binding(0) var<storage, read_write> keysDepths: array<f32>; // Need to represent the f32 bytes using u32 -> GPU just looks at it as 4 bytes
@group(1) @binding(1) var<storage, read_write> valuesIndices: array<u32>;
@group(1) @binding(2) var<storage, read_write> sortIndirect: SortIndirect;
@group(1) @binding(3) var<storage, read_write> sortInfos: GeneralInfo;

fn computeCov2D(mean: vec3<f32>, cov3D: array<f32, 6>) -> vec3<f32> {
    var t: vec4<f32> = uniforms.viewMatrix * vec4<f32>(mean, 1.0);

    let limx: f32 = 1.3 * uniforms.tanFovX;
    let limy: f32 = 1.3 * uniforms.tanFovY;
    let txtz: f32 = t.x / t.z;
    let tytz: f32 = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    let J: mat3x3<f32> = mat3x3<f32>(
        uniforms.focalX / t.z, 0.0, -(uniforms.focalX * t.x) / (t.z * t.z),
        0.0, uniforms.focalY / t.z, -(uniforms.focalY * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );

    let W: mat3x3<f32> = mat3x3<f32>(
        uniforms.viewMatrix[0][0], uniforms.viewMatrix[1][0], uniforms.viewMatrix[2][0],
        uniforms.viewMatrix[0][1], uniforms.viewMatrix[1][1], uniforms.viewMatrix[2][1],
        uniforms.viewMatrix[0][2], uniforms.viewMatrix[1][2], uniforms.viewMatrix[2][2]
    );

    let T: mat3x3<f32> = W * J;

    let Vrk: mat3x3<f32> = mat3x3<f32>(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]
    );

    var cov: mat3x3<f32> = transpose(T) * transpose(Vrk) * T;

    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

fn ndc2Pix(v: f32, S: f32) -> f32 {
    return ((v + 1.0) * S - 1.0) * 0.5;
}

@compute @workgroup_size(256, 1, 1)
fn preprocess(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let id: u32 = globalId.x;
    if (id >= arrayLength(&rawGaussians)) {
        return; // Because of ceiling
    }
    let rawGaussian: RawGaussian = rawGaussians[id];

    let pHom: vec4<f32> = uniforms.viewProjectionMatrix * vec4<f32>(rawGaussian.mean, 1.0);
    let pProj: vec3<f32> = pHom.xyz / (pHom.w + 0.0000001);

    let cov: vec3<f32> = computeCov2D(rawGaussian.mean, rawGaussian.cov3D);
    let det: f32 = cov.x * cov.z - cov.y * cov.y;
    let detInv: f32 = 1.0 / det; // Invert covariance (EWA algorithm)
    let conic: vec3<f32> = vec3<f32>(cov.z, -cov.y, cov.x) * detInv;

    // Compute extent in screen space (by finding eigenvalues of
    // 2D covariance matrix). Use extent to compute the bounding
    // rectangle of the splat in screen space

    let mid: f32 = 0.5 * (cov.x + cov.z);
    let lambda1: f32 = mid + sqrt(max(0.1, mid * mid - det)); // From https://www.johndcook.com/blog/2021/05/07/trick-for-2x2-eigenvalues/
    let lambda2: f32 = mid - sqrt(max(0.1, mid * mid - det));
    var radius: f32 = ceil(3.0 * sqrt(max(lambda1, lambda2))); // Longest axis (see https://cookierobotics.com/007/), 3.0 to ensure max Gaussian coverage
    let pointImage: vec2<f32> = vec2<f32>(ndc2Pix(pProj.x, uniforms.W), ndc2Pix(pProj.y, uniforms.H));

    let pView: vec4<f32> = uniforms.viewMatrix * vec4<f32>(rawGaussian.mean, 1.0);
    if (det == 0.0 || pView.z <= 0.4) {
        return;
    }

    let processedGaussian: ProcessedGaussian = ProcessedGaussian(
        pack2x16float(rawGaussian.color.rg), pack2x16float(vec2<f32>(rawGaussian.color.b, rawGaussian.opacity)),
        pack2x16float(conic.xy), pack2x16float(vec2<f32>(conic.z, radius)),
        pack2x16float(pointImage)
    );

    let storeId: u32 = atomicAdd(&sortInfos.num_keys, 1u);
    processedGaussians[storeId] = processedGaussian;
    keysDepths[storeId] = pHom.z;
    valuesIndices[storeId] = storeId;

    let keysPerWg = 256u * 15u;
    if (storeId % keysPerWg) == 0u {
        atomicAdd(&sortIndirect.dispatchX, 1u);
    }
}

// Radix sort lib used, draw indirect logic from:
// https://github.com/KeKsBoTer/wgpu_sort