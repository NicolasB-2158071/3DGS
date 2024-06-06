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
    meanOp: vec2<u32>,
    indexSH: u32,
    cov3D: array<u32, 3> // array (instead of vec3) to avoid unnecessary padding
}

struct ProcessedGaussian {
    colorRG: u32, colorBA: u32, // avoiding vec
    conicXY: u32, conicZR: u32,
    pointImage: u32
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> rawGaussians: array<RawGaussian>;
@group(0) @binding(2) var<storage, read_write> processedGaussians: array<ProcessedGaussian>;
@group(0) @binding(3) var<storage, read> shCoefficients: array<array<vec3<f32>, 16>>;

@group(1) @binding(0) var<storage, read_write> keysDepths: array<f32>; // Need to represent the f32 bytes using u32 -> GPU just looks at it as 4 bytes
@group(1) @binding(1) var<storage, read_write> valuesIndices: array<u32>;
@group(1) @binding(2) var<storage, read_write> sortIndirect: SortIndirect;
@group(1) @binding(3) var<storage, read_write> sortInfos: GeneralInfo;

const C0: f32 = 0.28209479177387814;
const C1: f32 = 0.4886025119029199;
const C2: array<f32, 5> = array(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const C3: array<f32, 7> = array(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

// Hardcode degree
fn evalSH(pos: vec3<f32>, sh: array<vec3<f32>, 16>) -> vec3<f32> {
    let dir = normalize(pos - uniforms.pos);
    var result = C0 * sh[0];

    // deg 0
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    result = result
        - C1 * y * sh[1]
        + C1 * z * sh[2]
        - C1 * x * sh[3];

    // deg 1
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    result = result +
        C2[0] * xy * sh[4] +
        C2[1] * yz * sh[5] +
        C2[2] * (2.0 * zz - xx - yy) * sh[6] +
        C2[3] * xz * sh[7] +
        C2[4] * (xx - yy) * sh[8];
    
    // deg 2
    result = result +
        C3[0] * y * (3.0 * xx - yy) * sh[9] +
        C3[1] * xy * z * sh[10] +
        C3[2] * y * (4.0 * zz - xx - yy) * sh[11] +
        C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12] +
        C3[4] * x * (4.0 * zz - xx - yy) * sh[13] +
        C3[5] * z * (xx - yy) * sh[14] +
        C3[6] * x * (xx - 3.0 * yy) * sh[15];

    result = result + 0.5;
    return result;
}

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
    let mo: vec4<f32> = vec4<f32>(unpack2x16float(rawGaussian.meanOp.x), unpack2x16float(rawGaussian.meanOp.y));
    let cova: vec2<f32> = unpack2x16float(rawGaussian.cov3D[0]);
    let covb: vec2<f32> = unpack2x16float(rawGaussian.cov3D[1]);
    let covc: vec2<f32> = unpack2x16float(rawGaussian.cov3D[2]);

    let cov3D: array<f32, 6> = array<f32, 6>(cova.x, cova.y, covb.x, covb.y, covc.x, covc.y);

    let mean: vec3<f32> = mo.xyz;
    let pHom: vec4<f32> = uniforms.viewProjectionMatrix * vec4<f32>(mean, 1.0);
    let pProj: vec3<f32> = pHom.xyz / (pHom.w + 0.0000001);

    let cov: vec3<f32> = computeCov2D(mean, cov3D);
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

    let pView: vec4<f32> = uniforms.viewMatrix * vec4<f32>(mean, 1.0);
    if (det == 0.0 || pView.z <= 0.4) {
        return;
    }

    let color: vec3<f32> = evalSH(mean, shCoefficients[rawGaussian.indexSH]);
    let processedGaussian: ProcessedGaussian = ProcessedGaussian(
        pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, mo.w)),
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