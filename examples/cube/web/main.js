import { createBridge } from '../../../lib/webgpu.js';

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;

async function main() {
    // Check WebGPU support
    if (!navigator.gpu) {
        document.body.innerHTML = '<h2>WebGPU is not supported in this browser.</h2><p>Try Chrome 113+ or Firefox 121+.</p>';
        return;
    }

    // Request adapter and device
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
        document.body.innerHTML = '<h2>No WebGPU adapter found.</h2>';
        return;
    }
    const device = await adapter.requestDevice();
    const queue = device.queue;

    // Set up canvas
    const canvas = document.getElementById('webgpu-canvas');
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    // Create bridge and pre-initialize with WebGPU objects
    const bridge = createBridge();
    bridge.setPreInitialized(adapter, device, queue, context, canvasFormat);

    // Map canvas format string to C API integer
    const FORMAT_MAP = {
        'rgba8unorm': 18, 'rgba8unorm-srgb': 19,
        'bgra8unorm': 23, 'bgra8unorm-srgb': 24,
    };
    const formatInt = FORMAT_MAP[canvasFormat] || 23;

    // Load WASM module
    const response = await fetch('cube_web.wasm');
    const { instance } = await WebAssembly.instantiateStreaming(response, { env: bridge.env });

    // Give bridge access to WASM memory
    bridge.setMemory(instance.exports.memory);

    // Call _initialize if present (C3 runtime init for globals/statics)
    if (instance.exports._initialize) {
        instance.exports._initialize();
    }

    // Initialize the cube
    instance.exports.init(CANVAS_WIDTH, CANVAS_HEIGHT, formatInt);

    // Render loop
    function frame() {
        instance.exports.render_frame();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    console.log('WebGPU cube running!');
}

main().catch(err => {
    console.error('Fatal error:', err);
    document.body.innerHTML = `<h2>Error</h2><pre>${err.message}\n${err.stack}</pre>`;
});
