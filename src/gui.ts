import GUI from "lil-gui";

export const gui: GUI = new GUI({title: "Settings"});

export function initGUI(settings: Object) {
    gui.addColor(settings, "backgroundColor").name("Background color");

    const gaussianFolder = gui.addFolder("Gaussian controls");
    gaussianFolder.add(settings, "scalingModifier", 0.01, 1, 0.01).name("Scaling modifier");
    gaussianFolder.add(settings, "uploadDataFile").name("Upload data file");

    const cameraFolder = gui.addFolder("Camera controls");
    cameraFolder.add(settings, "cameraSpeed", 0.0, 1.0, 0.1).name("Camera speed");
    cameraFolder.add(settings, "uploadCameraJson").name("Upload camera file");

    const timingFolder = gui.addFolder("Timing");
    timingFolder.add(settings, "fps").listen().disable();
    timingFolder.add(settings, "preprocessTime").name("preprocess time (ms)").listen().disable();
    timingFolder.add(settings, "sortTime").name("sort time (ms)").listen().disable();
    timingFolder.add(settings, "renderTime").name("render time (ms)").listen().disable();
}