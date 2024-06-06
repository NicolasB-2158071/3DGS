export class Timer {
    private querySet: GPUQuerySet;
    private queryBuffer: GPUBuffer;
    private resultBuffer: GPUBuffer;

    private subscribed: Map<string, number>;
    private lastIndex: number;
    private times: Array<number>;

    constructor(device: GPUDevice) {
        this.querySet = device.createQuerySet({
            type: "timestamp",
            count: 10 // Hardcoded
        });
        this.queryBuffer = device.createBuffer({
            size: this.querySet.count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
        });
        this.resultBuffer = device.createBuffer({
            size: this.queryBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        this.lastIndex = 0;
        this.subscribed = new Map<string, number>();
        this.times = new Array<number>(10).fill(0);
    }

    public subscribe(name: string): void {
        this.subscribed.set(name, this.lastIndex);
        this.lastIndex += 2;
    }

    public getSubscription(name: string): GPUComputePassTimestampWrites {
        return {
            querySet: this.querySet,
            beginningOfPassWriteIndex: this.subscribed.get(name),
            endOfPassWriteIndex: this.subscribed.get(name) + 1
        }
    }

    public submitTimestamps(encoder: GPUCommandEncoder): void {
        encoder.resolveQuerySet(this.querySet, 0, 10, this.queryBuffer, 0);
        if (this.resultBuffer.mapState === "unmapped") {
            encoder.copyBufferToBuffer(this.queryBuffer, 0, this.resultBuffer, 0, this.resultBuffer.size);
        }
    }

    public resolveResults(): void {
        if (this.resultBuffer.mapState === "unmapped") {
            this.resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
                let times: BigInt64Array = new BigInt64Array(this.resultBuffer.getMappedRange());
                times.forEach((value: bigint, index: number) => {
                    this.times[index] = Number(value);
                });
                this.resultBuffer.unmap();
            });
        }
    }

    public getResultByName(name: string): number {
        let id: number = this.subscribed.get(name);
        return Number(Number((this.times[id + 1] - this.times[id]) / 1000000).toFixed(1));
    }

    public getResultByIds(idOne: number, idTwo: number): number {
        return Number(Number((this.times[idOne + 1] - this.times[idTwo]) / 1000000).toFixed(1));
    }

    public getIdOfName(name: string): number {
        return this.subscribed.get(name);
    }
}