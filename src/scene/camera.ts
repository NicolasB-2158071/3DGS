import { Mat3, Mat4, Vec3, mat4, vec3 } from "wgpu-matrix";
import { degreeToRad, radToDegree, invertRow } from "../utils/maths";

export interface CameraViewPoint {
    name: string,
    rotation: Mat3,
    position: Vec3
}

export class Camera {
    public tanFovX: number;
    public tanFovY: number;
    public focalX: number;
    public focalY: number;
    public FOVY: number = 0.820176;

    private pos: Vec3;
    private front: Vec3;
    private up: Vec3;
    private speed: number;

    private viewMatrix: Mat4;
    private projectionMatrix: Mat4;
    private VPCopy: Mat4;

    private keyStates: any;
    private isDragging: boolean;

    private lastX: number;
    private lastY: number;
    private yaw: number;
    private pitch: number;

    constructor(canvasWidth: number, canvasHeight: number) {
        this.initCameraRatios(canvasWidth, canvasHeight);

        this.pos = vec3.create(0, 0, 0);
        this.front = vec3.create(0, 0, 1.0);
        this.up = vec3.create(0, 1.0, 0.0);
        this.speed = 0.07;

        this.viewMatrix = mat4.create();
        this.projectionMatrix = mat4.create();
        this.initProjectionMatrix(canvasWidth, canvasHeight);
        this.VPCopy = mat4.create();

        this.keyStates = {
            KeyW: false,
            KeyS: false,
            KeyA: false,
            KeyD: false,
        };
        this.isDragging = false;
        this.lastX = 0;
        this.lastY = 0;
        this.yaw = 90.0;
        this.pitch = 0.0;

        this.createCallBacks()
    }

    public initProjectionMatrix(canvasWidth: number, canvasHeight: number) {
        this.initCameraRatios(canvasWidth, canvasHeight);
        mat4.perspective(this.FOVY, canvasWidth / canvasHeight, 0.1, 100, this.projectionMatrix);
    }

    private initCameraRatios(width: number, height: number) {
        this.tanFovY = Math.tan(this.FOVY * 0.5);
        this.tanFovX = this.tanFovY * width / height;
        this.focalY = height / (2.0 * this.tanFovY);
        this.focalX = width / (2.0 * this.tanFovX);
    }

    public update(speed: number): boolean {
        this.speed = speed;
        let right: Vec3 = vec3.normalize(vec3.cross(this.front, this.up));

        if (this.keyStates.KeyW) vec3.add(this.pos, vec3.scale(this.front, this.speed), this.pos);
        if (this.keyStates.KeyS) vec3.subtract(this.pos, vec3.scale(this.front, this.speed), this.pos);
        if (this.keyStates.KeyD) vec3.subtract(this.pos, vec3.scale(right, this.speed), this.pos);
        if (this.keyStates.KeyA) vec3.add(this.pos, vec3.scale(right, this.speed), this.pos);

        mat4.lookAt(this.pos, vec3.add(this.pos, this.front), this.up, this.viewMatrix);

        let vp: Mat4 = mat4.multiply(this.projectionMatrix, this.viewMatrix);
        if (mat4.equals(vp, this.VPCopy)) {return false;}
        mat4.copy(vp, this.VPCopy);
        return true;
    }

    public getViewMatrix(): Mat4 {
        let cp: Mat4 = mat4.copy(this.viewMatrix); // Shallow copy
        invertRow(cp, 1); // Target system is different (see sibr viewer, gitlab implementation)
        invertRow(cp, 2);
        invertRow(cp, 0);

        return cp;
    }

    public getVP(): Mat4 {
        let vp: Mat4 = mat4.multiply(this.projectionMatrix, this.viewMatrix);
        invertRow(vp, 1);
        invertRow(vp, 0);

        return vp;
    }

    public getPosition(): Vec3 {
        return this.pos;
    }

    public setViewPoint(viewPoint: CameraViewPoint): void {
        let vm: Mat4 = mat4.fromMat3(viewPoint.rotation);
        mat4.translate(vm, vec3.mulScalar(viewPoint.position, -1), vm);

        this.pos = viewPoint.position;
        vec3.normalize(vec3.fromValues(vm[2], vm[6], vm[10]), this.front);
        vec3.normalize(vec3.fromValues(vm[1], vm[5], vm[9]), this.up);

        this.yaw = radToDegree(Math.atan(this.front[2] / this.front[0]));
        this.pitch = radToDegree(Math.asin(this.front[1]));

        this.yaw = (this.yaw < 0) ? (this.yaw + 180) : this.yaw;
        // reduction pitch to I or IV quadrant?
    }

    private createCallBacks() {
        document.addEventListener("keydown", e => {
            if (this.keyStates[e.code] == null)
                return;
            this.keyStates[e.code] = true;
        });
        document.addEventListener("keyup", e => {
            if (this.keyStates[e.code] == null) 
                return;
            this.keyStates[e.code] = false;
        });
        document.addEventListener("mousedown", e => {
            this.isDragging = true;
            this.lastX = e.pageX;
            this.lastY = e.pageY;
        });
        document.addEventListener("mouseup", e => {
            this.isDragging = false;

        });
        document.addEventListener("mousemove", e => {
            if (!this.isDragging) return;
            let offsetX: number = e.pageX - this.lastX;
            let offsetY: number = e.pageY - this.lastY;
            this.lastX = e.pageX; this.lastY = e.pageY;
            this.yaw += offsetX * this.speed; this.pitch -= offsetY * this.speed;
        
            if (this.pitch > 89.0)
                this.pitch = 89.0;
            if (this.pitch < -89.0)
                this.pitch = -89.0;
        
            this.front = vec3.normalize(vec3.create(
                Math.cos(degreeToRad(this.yaw)) * Math.cos(degreeToRad(this.pitch)),
                Math.sin(degreeToRad(this.pitch)),
                Math.sin(degreeToRad(this.yaw)) * Math.cos(degreeToRad(this.pitch))
            ));
        });
    }
}