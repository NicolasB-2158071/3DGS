export function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

import { Vec3, Vec4, Mat3, mat3, Mat4 } from "wgpu-matrix";
export function computeCov3D(scale: Vec3, quaternion: Vec4, scale_modifier: number = 1): Array<number> {
    let S: Mat3 = mat3.create();
    let R: Mat3 = mat3.create();
    let M: Mat3 = mat3.create();
    let Sigma: Mat3 = mat3.create();

    mat3.set(
        scale_modifier * scale[0], 0, 0,
        0, scale_modifier * scale[1], 0,
        0, 0, scale_modifier * scale[2],
        S
    );

    const r: number = quaternion[0];
    const x: number = quaternion[1];
    const y: number = quaternion[2];
    const z: number = quaternion[3];

    // Quaternion matrix -> rotation matrix
    mat3.set(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1. - 2. * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y),
        R
    );
    mat3.multiply(S, R, M);  // M = S * R

    // Compute 3D world covariance matrix Sigma
    mat3.multiply(mat3.transpose(M), M, Sigma); // Sigma = transpose(M) * M

    // Covariance is symmetric, only store upper right
    const cov3D: Array<number> = [
        Sigma[0], Sigma[1], Sigma[2],
        Sigma[5], Sigma[6], Sigma[10] // Mat3 has padding in wgpu matrix
    ];

    return cov3D;
}

export function normalizeQuaternion(rotation: Float32Array): Float32Array {
    let euclidianNorm: number = 0.0;
    for (let i = 0; i < rotation.length; ++i) // Avoid copy of rotation
        euclidianNorm += (rotation[i] * rotation[i]);
    euclidianNorm = Math.sqrt(euclidianNorm);

    return rotation.map(x => x / euclidianNorm);
}

export function expArray(array: Float32Array): Float32Array {
    return array.map((x) => Math.exp(x)); // ply file takes log of scale
}

export function degreeToRad(degree: number) {
    return (degree * Math.PI) / 180.0;
}

export function radToDegree(radians: number) {
    return radians * 180 / Math.PI;
}

export function invertRow(mat: Mat4, row: number): void {
    mat[row + 0] = -mat[row + 0];
    mat[row + 4] = -mat[row + 4];
    mat[row + 8] = -mat[row + 8];
    mat[row + 12] = -mat[row + 12];
}