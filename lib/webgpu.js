// WebGPU JavaScript bridge for C3 WASM
// Maps wgpuXXX C API calls to browser WebGPU JavaScript API

// Handle table: integer handle <-> JavaScript WebGPU object
let nextHandle = 1;
const handles = new Map();
function addHandle(obj) { const h = nextHandle++; handles.set(h, obj); return h; }
function getHandle(h) { return handles.get(h); }

// Enum mappings: C API integer constants -> browser WebGPU strings
const TEXTURE_FORMAT = {
    18: 'rgba8unorm', 19: 'rgba8unorm-srgb',
    23: 'bgra8unorm', 24: 'bgra8unorm-srgb',
    45: 'depth16unorm', 46: 'depth24plus', 47: 'depth24plus-stencil8',
    48: 'depth32float', 49: 'depth32float-stencil8',
};
const TEXTURE_FORMAT_REV = {};
for (const [k, v] of Object.entries(TEXTURE_FORMAT)) TEXTURE_FORMAT_REV[v] = Number(k);

const PRIMITIVE_TOPOLOGY = { 1: 'point-list', 2: 'line-list', 3: 'line-strip', 4: 'triangle-list', 5: 'triangle-strip' };
const FRONT_FACE = { 1: 'ccw', 2: 'cw' };
const CULL_MODE = { 0: undefined, 1: 'none', 2: 'front', 3: 'back' };
const LOAD_OP = { 1: 'load', 2: 'clear' };
const STORE_OP = { 1: 'store', 2: 'discard' };
const INDEX_FORMAT = { 1: 'uint16', 2: 'uint32' };
const VERTEX_STEP_MODE = { 0: undefined, 1: undefined, 2: 'vertex', 3: 'instance' };
const BUFFER_BINDING_TYPE = { 0: undefined, 1: undefined, 2: 'uniform', 3: 'storage', 4: 'read-only-storage' };
const COMPARE_FUNCTION = { 1: 'never', 2: 'less', 3: 'equal', 4: 'less-equal', 5: 'greater', 6: 'not-equal', 7: 'greater-equal', 8: 'always' };
const TEXTURE_DIMENSION = { 1: '1d', 2: '2d', 3: '3d' };

const VERTEX_FORMAT = {
    1: 'uint8', 2: 'uint8x2', 3: 'uint8x4',
    4: 'sint8', 5: 'sint8x2', 6: 'sint8x4',
    7: 'unorm8', 8: 'unorm8x2', 9: 'unorm8x4',
    10: 'snorm8', 11: 'snorm8x2', 12: 'snorm8x4',
    13: 'uint16', 14: 'uint16x2', 15: 'uint16x4',
    16: 'sint16', 17: 'sint16x2', 18: 'sint16x4',
    19: 'unorm16', 20: 'unorm16x2', 21: 'unorm16x4',
    22: 'snorm16', 23: 'snorm16x2', 24: 'snorm16x4',
    25: 'float16', 26: 'float16x2', 27: 'float16x4',
    28: 'float32', 29: 'float32x2', 30: 'float32x3', 31: 'float32x4',
    32: 'uint32', 33: 'uint32x2', 34: 'uint32x3', 35: 'uint32x4',
    36: 'sint32', 37: 'sint32x2', 38: 'sint32x3', 39: 'sint32x4',
};

// Pre-initialized WebGPU objects (set before WASM load)
let preAdapter = null;
let preDevice = null;
let preQueue = null;
let preCanvasContext = null;
let preCanvasFormat = null;

// WASM memory access helpers
let wasmMemory = null;

function dv() { return new DataView(wasmMemory.buffer); }
function u8() { return new Uint8Array(wasmMemory.buffer); }
function u32() { return new Uint32Array(wasmMemory.buffer); }

function readU32(ptr) { return dv().getUint32(ptr, true); }
function readI32(ptr) { return dv().getInt32(ptr, true); }
function readU64(ptr) { return Number(dv().getBigUint64(ptr, true)); }
function readF64(ptr) { return dv().getFloat64(ptr, true); }
function writeU32(ptr, val) { dv().setUint32(ptr, val, true); }

function readStringView(ptr) {
    // StringView: { char* data: +0, usz length: +4 } (8 bytes on wasm32)
    const dataPtr = readU32(ptr);
    const length = readU32(ptr + 4);
    if (dataPtr === 0 && length === 0) return '';
    const bytes = u8();
    if (length === 0xFFFFFFFF) {
        // WGPU_STRLEN: null-terminated
        let end = dataPtr;
        while (bytes[end] !== 0) end++;
        return new TextDecoder().decode(bytes.slice(dataPtr, end));
    }
    return new TextDecoder().decode(bytes.slice(dataPtr, dataPtr + length));
}

// Struct layout offsets for wasm32
// Computed from C3 struct definitions with: ptr=4, uint=4, ulong=8, usz=4
const OFF = {
    // BufferDescriptor (40 bytes)
    // +0: void* next_in_chain, +4: StringView label (8), +12: pad4, +16: BufferUsage(u64), +24: ulong size, +32: uint mapped
    BufferDesc: { usage: 16, size: 24, mapped: 32 },

    // ShaderModuleDescriptor (12 bytes)
    // +0: void* next_in_chain, +4: StringView label
    ShaderModuleDesc: { next_in_chain: 0, label: 4 },

    // ShaderSourceWGSL (16 bytes)
    // +0: void* chain_next, +4: uint s_type, +8: StringView code
    ShaderSourceWGSL: { chain_next: 0, s_type: 4, code: 8 },

    // SurfaceConfiguration (48 bytes)
    // +0: void* next, +4: Device, +8: format(u32), +12: pad4, +16: TextureUsage(u64),
    // +24: width, +28: height, +32: view_formats_count, +36: view_formats*, +40: alpha_mode, +44: present_mode
    SurfaceConfig: { device: 4, format: 8, usage: 16, width: 24, height: 28 },

    // BindGroupLayoutEntry (80 bytes)
    // +0: void* next, +4: uint binding, +8: ShaderStage visibility(u64),
    // +16: BufferBindingLayout buffer { +16: void* next, +20: type(u32), +24: has_dynamic_offset, +28: pad4, +32: min_binding_size(u64) }
    BGLEntry: { binding: 4, visibility: 8, buffer_type: 20, buffer_has_dynamic: 24, buffer_min_size: 32 },

    // BindGroupLayoutDescriptor (20 bytes)
    // +0: void* next, +4: StringView label (8), +12: usz entries_count, +16: entries*
    BGLDesc: { label: 4, entries_count: 12, entries: 16 },

    // BindGroupEntry (40 bytes)
    // +0: void* next, +4: uint binding, +8: Buffer, +12: pad4, +16: ulong offset, +24: ulong size, +32: Sampler, +36: TextureView
    BGEntry: { binding: 4, buffer: 8, offset: 16, size: 24, sampler: 32, texture_view: 36 },

    // BindGroupDescriptor (24 bytes)
    // +0: void* next, +4: StringView label (8), +12: BindGroupLayout layout, +16: usz entries_count, +20: entries*
    BGDesc: { label: 4, layout: 12, entries_count: 16, entries: 20 },

    // PipelineLayoutDescriptor (24 bytes)
    // +0: void* next, +4: StringView label (8), +12: usz count, +16: layouts*, +20: uint immediate_size
    PLDesc: { label: 4, count: 12, layouts: 16 },

    // VertexAttribute (24 bytes)
    // +0: uint format, +4: pad4, +8: ulong offset, +16: uint shader_location, +20: pad4
    VertexAttr: { format: 0, offset: 8, shader_location: 16, SIZEOF: 24 },

    // VertexBufferLayout (24 bytes)
    // +0: uint step_mode, +4: pad4, +8: ulong array_stride, +16: usz attr_count, +20: attrs*
    VBLayout: { step_mode: 0, array_stride: 8, attr_count: 16, attrs: 20, SIZEOF: 24 },

    // VertexState (32 bytes)
    // +0: void* next, +4: ShaderModule, +8: StringView entry_point (8), +16: usz const_count, +20: consts*, +24: usz buf_count, +28: bufs*
    VertexState: { module: 4, entry_point: 8, const_count: 16, consts: 20, buf_count: 24, bufs: 28 },

    // FragmentState (32 bytes) - same layout as VertexState but last fields differ
    // +0: void* next, +4: ShaderModule, +8: StringView entry_point (8), +16: usz const_count, +20: consts*, +24: usz target_count, +28: targets*
    FragmentState: { module: 4, entry_point: 8, target_count: 24, targets: 28 },

    // ColorTargetState (24 bytes)
    // +0: void* next, +4: uint format, +8: BlendState* blend, +12: pad4, +16: ColorWriteMask(u64)
    ColorTarget: { format: 4, blend: 8, write_mask: 16, SIZEOF: 24 },

    // PrimitiveState (24 bytes)
    // +0: void* next, +4: topology, +8: strip_index_format, +12: front_face, +16: cull_mode, +20: unclipped_depth
    PrimState: { topology: 4, strip_format: 8, front_face: 12, cull_mode: 16 },

    // MultisampleState (16 bytes)
    // +0: void* next, +4: count, +8: mask, +12: alpha_to_coverage
    MSState: { count: 4, mask: 8 },

    // RenderPipelineDescriptor (96 bytes)
    // +0: void* next, +4: StringView label (8), +12: PipelineLayout layout,
    // +16: VertexState vertex (32 bytes), +48: PrimitiveState primitive (24 bytes),
    // +72: DepthStencilState* depth_stencil, +76: MultisampleState multisample (16 bytes),
    // +92: FragmentState* fragment
    RPDesc: { label: 4, layout: 12, vertex: 16, primitive: 48, depth_stencil: 72, multisample: 76, fragment: 92 },

    // RenderPassColorAttachment (56 bytes)
    // +0: void* next, +4: TextureView view, +8: uint depth_slice, +12: TextureView resolve,
    // +16: uint load_op, +20: uint store_op, +24: Color clear_value (4x double = 32 bytes)
    RPColorAtt: { view: 4, depth_slice: 8, resolve: 12, load_op: 16, store_op: 20, clear: 24, SIZEOF: 56 },

    // RenderPassDescriptor (32 bytes)
    // +0: void* next, +4: StringView label (8), +12: usz ca_count, +16: ca*, +20: ds*, +24: occlusion_qs, +28: timestamp*
    RPDesc2: { label: 4, ca_count: 12, ca: 16, ds: 20 },

    // TextureDescriptor offsets (wasm32)
    // +0: next_in_chain (ptr), +4: StringView (8), +12: pad4, +16: TextureUsage (u64),
    // +24: dimension (u32), +28: Extent3D {w,h,d} (12), +40: format (u32),
    // +44: mip_level_count (u32), +48: sample_count (u32)
    TexDesc: { usage: 16, dimension: 24, width: 28, height: 32, depth: 36, format: 40, mip: 44, samples: 48 },

    // DepthStencilState offsets
    DSState: { format: 4, depth_write: 8, depth_compare: 12 },

    // RenderPassDepthStencilAttachment offsets
    RPDSAtt: { view: 4, depth_load_op: 8, depth_store_op: 12, depth_clear_value: 16 },

    // SurfaceTexture (12 bytes)
    // +0: void* next, +4: Texture, +8: uint status
    SurfTex: { next: 0, texture: 4, status: 8 },

    // RequestAdapterCallbackInfo (20 bytes)
    // +0: void* next, +4: uint mode, +8: callback fn ptr, +12: void* userdata1, +16: void* userdata2
    ReqAdapterCB: { callback: 8, userdata1: 12 },

    // RequestDeviceCallbackInfo (20 bytes)
    ReqDeviceCB: { callback: 8, userdata1: 12 },

    // RequestAdapterOptions (24 bytes)
    // +0: void* next, +4: feature_level, +8: power_preference, +12: force_fallback, +16: backend_type, +20: compatible_surface
    ReqAdapterOpts: { feature_level: 4, power_preference: 8 },

    // CommandEncoderDescriptor (12 bytes), CommandBufferDescriptor (12 bytes)
    // +0: void* next, +4: StringView label
    CmdEncDesc: { label: 4 },
    CmdBufDesc: { label: 4 },

    // SurfaceDescriptor (12 bytes)
    // +0: void* next, +4: StringView label
    SurfDesc: { label: 4 },
};

export function createBridge() {
    const env = {
        // (i32 desc_ptr) -> i32 instance_handle
        wgpuCreateInstance(descPtr) {
            return addHandle({ type: 'instance' });
        },

        // (i32 instance, i32 opts_ptr, i32 cb_info_ptr) -> i64 Future
        wgpuInstanceRequestAdapter(instance, optsPtr, cbInfoPtr) {
            // Write pre-obtained adapter handle directly to callback's userdata1 target
            const ud1 = readU32(cbInfoPtr + OFF.ReqAdapterCB.userdata1);
            const adapterHandle = addHandle(preAdapter);
            writeU32(ud1, adapterHandle);
            return 0n; // Future.id
        },

        // (i32 adapter, i32 desc_ptr, i32 cb_info_ptr) -> i64 Future
        wgpuAdapterRequestDevice(adapter, descPtr, cbInfoPtr) {
            const ud1 = readU32(cbInfoPtr + OFF.ReqDeviceCB.userdata1);
            const deviceHandle = addHandle(preDevice);
            writeU32(ud1, deviceHandle);
            return 0n;
        },

        // (i32 device) -> i32 queue
        wgpuDeviceGetQueue(device) {
            return addHandle(preQueue);
        },

        // (i32 instance, i32 desc_ptr) -> i32 surface
        wgpuInstanceCreateSurface(instance, descPtr) {
            return addHandle({ type: 'surface', context: preCanvasContext });
        },

        // (i32 surface, i32 config_ptr) -> void
        wgpuSurfaceConfigure(surfaceHandle, configPtr) {
            const surface = getHandle(surfaceHandle);
            const format = TEXTURE_FORMAT[readU32(configPtr + OFF.SurfaceConfig.format)] || preCanvasFormat;
            const usage = readU32(configPtr + OFF.SurfaceConfig.usage); // low 32 bits of ulong
            surface.context.configure({
                device: preDevice,
                format: format,
                usage: usage,
                alphaMode: 'opaque',
            });
        },

        // (i32 device, i32 desc_ptr) -> i32 buffer
        wgpuDeviceCreateBuffer(deviceHandle, descPtr) {
            const device = getHandle(deviceHandle);
            const usage = readU32(descPtr + OFF.BufferDesc.usage); // low 32 bits
            const size = readU64(descPtr + OFF.BufferDesc.size);
            const mapped = readU32(descPtr + OFF.BufferDesc.mapped);
            const buf = device.createBuffer({
                size: size,
                usage: usage,
                mappedAtCreation: !!mapped,
            });
            return addHandle(buf);
        },

        // (i32 queue, i32 buffer, i64 offset, i32 data_ptr, i32 size) -> void
        wgpuQueueWriteBuffer(queueHandle, bufferHandle, offset, dataPtr, size) {
            const queue = getHandle(queueHandle);
            const buffer = getHandle(bufferHandle);
            const off = Number(offset);
            const data = new Uint8Array(wasmMemory.buffer, dataPtr, size);
            queue.writeBuffer(buffer, off, data);
        },

        // (i32 device, i32 desc_ptr) -> i32 bind_group_layout
        wgpuDeviceCreateBindGroupLayout(deviceHandle, descPtr) {
            const device = getHandle(deviceHandle);
            const count = readU32(descPtr + OFF.BGLDesc.entries_count);
            const entriesPtr = readU32(descPtr + OFF.BGLDesc.entries);
            const entries = [];
            for (let i = 0; i < count; i++) {
                const ePtr = entriesPtr + i * 80; // BindGroupLayoutEntry is 80 bytes
                const binding = readU32(ePtr + OFF.BGLEntry.binding);
                const visibility = readU32(ePtr + OFF.BGLEntry.visibility); // low 32 of ulong
                const bufType = readU32(ePtr + OFF.BGLEntry.buffer_type);
                const entry = { binding, visibility };
                if (bufType && BUFFER_BINDING_TYPE[bufType]) {
                    entry.buffer = {
                        type: BUFFER_BINDING_TYPE[bufType],
                        hasDynamicOffset: !!readU32(ePtr + OFF.BGLEntry.buffer_has_dynamic),
                        minBindingSize: readU64(ePtr + OFF.BGLEntry.buffer_min_size),
                    };
                }
                entries.push(entry);
            }
            return addHandle(device.createBindGroupLayout({ entries }));
        },

        // (i32 device, i32 desc_ptr) -> i32 pipeline_layout
        wgpuDeviceCreatePipelineLayout(deviceHandle, descPtr) {
            const device = getHandle(deviceHandle);
            const count = readU32(descPtr + OFF.PLDesc.count);
            const layoutsPtr = readU32(descPtr + OFF.PLDesc.layouts);
            const bindGroupLayouts = [];
            for (let i = 0; i < count; i++) {
                bindGroupLayouts.push(getHandle(readU32(layoutsPtr + i * 4)));
            }
            return addHandle(device.createPipelineLayout({ bindGroupLayouts }));
        },

        // (i32 device, i32 desc_ptr) -> i32 bind_group
        wgpuDeviceCreateBindGroup(deviceHandle, descPtr) {
            const device = getHandle(deviceHandle);
            const layout = getHandle(readU32(descPtr + OFF.BGDesc.layout));
            const count = readU32(descPtr + OFF.BGDesc.entries_count);
            const entriesPtr = readU32(descPtr + OFF.BGDesc.entries);
            const entries = [];
            for (let i = 0; i < count; i++) {
                const ePtr = entriesPtr + i * 40; // BindGroupEntry is 40 bytes
                const binding = readU32(ePtr + OFF.BGEntry.binding);
                const bufHandle = readU32(ePtr + OFF.BGEntry.buffer);
                const offset = readU64(ePtr + OFF.BGEntry.offset);
                const size = readU64(ePtr + OFF.BGEntry.size);
                if (bufHandle) {
                    entries.push({
                        binding,
                        resource: { buffer: getHandle(bufHandle), offset, size },
                    });
                }
            }
            return addHandle(device.createBindGroup({ layout, entries }));
        },

        // (i32 device, i32 desc_ptr) -> i32 shader_module
        wgpuDeviceCreateShaderModule(deviceHandle, descPtr) {
            const device = getHandle(deviceHandle);
            // Follow next_in_chain pointer to get ShaderSourceWGSL
            const chainPtr = readU32(descPtr + OFF.ShaderModuleDesc.next_in_chain);
            if (chainPtr === 0) throw new Error('ShaderModule: no source chain');
            const sType = readU32(chainPtr + OFF.ShaderSourceWGSL.s_type);
            if (sType !== 2) throw new Error('ShaderModule: expected WGSL source (s_type=2), got ' + sType);
            const code = readStringView(chainPtr + OFF.ShaderSourceWGSL.code);
            return addHandle(device.createShaderModule({ code }));
        },

        // (i32 device, i32 desc_ptr) -> i32 render_pipeline
        wgpuDeviceCreateRenderPipeline(deviceHandle, descPtr) {
            const device = getHandle(deviceHandle);
            const layout = getHandle(readU32(descPtr + OFF.RPDesc.layout));

            // Read VertexState (inline at +16, 32 bytes)
            const vsBase = descPtr + OFF.RPDesc.vertex;
            const vsModule = getHandle(readU32(vsBase + OFF.VertexState.module));
            const vsEntry = readStringView(vsBase + OFF.VertexState.entry_point);
            const vsBufCount = readU32(vsBase + OFF.VertexState.buf_count);
            const vsBufsPtr = readU32(vsBase + OFF.VertexState.bufs);
            const buffers = [];
            for (let i = 0; i < vsBufCount; i++) {
                const bPtr = vsBufsPtr + i * OFF.VBLayout.SIZEOF;
                const stepMode = VERTEX_STEP_MODE[readU32(bPtr + OFF.VBLayout.step_mode)] || 'vertex';
                const arrayStride = readU64(bPtr + OFF.VBLayout.array_stride);
                const attrCount = readU32(bPtr + OFF.VBLayout.attr_count);
                const attrsPtr = readU32(bPtr + OFF.VBLayout.attrs);
                const attributes = [];
                for (let j = 0; j < attrCount; j++) {
                    const aPtr = attrsPtr + j * OFF.VertexAttr.SIZEOF;
                    attributes.push({
                        format: VERTEX_FORMAT[readU32(aPtr + OFF.VertexAttr.format)],
                        offset: readU64(aPtr + OFF.VertexAttr.offset),
                        shaderLocation: readU32(aPtr + OFF.VertexAttr.shader_location),
                    });
                }
                buffers.push({ arrayStride, stepMode, attributes });
            }

            // Read PrimitiveState (inline at +48, 24 bytes)
            const psBase = descPtr + OFF.RPDesc.primitive;
            const primitive = {
                topology: PRIMITIVE_TOPOLOGY[readU32(psBase + OFF.PrimState.topology)],
                frontFace: FRONT_FACE[readU32(psBase + OFF.PrimState.front_face)],
                cullMode: CULL_MODE[readU32(psBase + OFF.PrimState.cull_mode)],
            };

            // Read MultisampleState (inline at +76, 16 bytes)
            const msBase = descPtr + OFF.RPDesc.multisample;
            const multisample = {
                count: readU32(msBase + OFF.MSState.count),
                mask: readU32(msBase + OFF.MSState.mask),
            };

            // Read FragmentState (pointer at +92)
            let fragment = undefined;
            const fragPtr = readU32(descPtr + OFF.RPDesc.fragment);
            if (fragPtr) {
                const fsModule = getHandle(readU32(fragPtr + OFF.FragmentState.module));
                const fsEntry = readStringView(fragPtr + OFF.FragmentState.entry_point);
                const fsTargetCount = readU32(fragPtr + OFF.FragmentState.target_count);
                const fsTargetsPtr = readU32(fragPtr + OFF.FragmentState.targets);
                const targets = [];
                for (let i = 0; i < fsTargetCount; i++) {
                    const tPtr = fsTargetsPtr + i * OFF.ColorTarget.SIZEOF;
                    const format = TEXTURE_FORMAT[readU32(tPtr + OFF.ColorTarget.format)] || preCanvasFormat;
                    const writeMask = readU32(tPtr + OFF.ColorTarget.write_mask); // low 32 of ulong
                    targets.push({ format, writeMask });
                }
                fragment = { module: fsModule, entryPoint: fsEntry, targets };
            }

            // Read DepthStencilState (pointer at +72)
            let depthStencil = undefined;
            const dsPtr = readU32(descPtr + OFF.RPDesc.depth_stencil);
            if (dsPtr) {
                depthStencil = {
                    format: TEXTURE_FORMAT[readU32(dsPtr + OFF.DSState.format)],
                    depthWriteEnabled: readU32(dsPtr + OFF.DSState.depth_write) === 1,
                    depthCompare: COMPARE_FUNCTION[readU32(dsPtr + OFF.DSState.depth_compare)],
                };
            }

            const desc = {
                layout,
                vertex: { module: vsModule, entryPoint: vsEntry, buffers },
                primitive,
                multisample,
            };
            if (depthStencil) desc.depthStencil = depthStencil;
            if (fragment) desc.fragment = fragment;

            return addHandle(device.createRenderPipeline(desc));
        },

        // (i32 device, i32 desc_ptr) -> i32 texture
        wgpuDeviceCreateTexture(deviceHandle, descPtr) {
            const device = getHandle(deviceHandle);
            const usage = readU32(descPtr + OFF.TexDesc.usage);
            const dimension = TEXTURE_DIMENSION[readU32(descPtr + OFF.TexDesc.dimension)] || '2d';
            const width = readU32(descPtr + OFF.TexDesc.width);
            const height = readU32(descPtr + OFF.TexDesc.height);
            const depthOrArrayLayers = readU32(descPtr + OFF.TexDesc.depth);
            const format = TEXTURE_FORMAT[readU32(descPtr + OFF.TexDesc.format)];
            const mipLevelCount = readU32(descPtr + OFF.TexDesc.mip);
            const sampleCount = readU32(descPtr + OFF.TexDesc.samples);
            const tex = device.createTexture({
                size: { width, height, depthOrArrayLayers },
                format, usage, dimension, mipLevelCount, sampleCount,
            });
            return addHandle(tex);
        },

        // (i32 surface, i32 out_ptr) -> void
        wgpuSurfaceGetCurrentTexture(surfaceHandle, outPtr) {
            const surface = getHandle(surfaceHandle);
            const texture = surface.context.getCurrentTexture();
            const texHandle = addHandle(texture);
            // Write SurfaceTexture: { next_in_chain: 0, texture: handle, status: 1 (SUCCESS_OPTIMAL) }
            writeU32(outPtr + OFF.SurfTex.next, 0);
            writeU32(outPtr + OFF.SurfTex.texture, texHandle);
            writeU32(outPtr + OFF.SurfTex.status, 1); // SUCCESS_OPTIMAL
        },

        // (i32 texture, i32 desc_ptr) -> i32 texture_view
        wgpuTextureCreateView(textureHandle, descPtr) {
            const texture = getHandle(textureHandle);
            return addHandle(texture.createView());
        },

        // (i32 device, i32 desc_ptr) -> i32 command_encoder
        wgpuDeviceCreateCommandEncoder(deviceHandle, descPtr) {
            const device = getHandle(deviceHandle);
            return addHandle(device.createCommandEncoder());
        },

        // (i32 encoder, i32 desc_ptr) -> i32 render_pass
        wgpuCommandEncoderBeginRenderPass(encoderHandle, descPtr) {
            const encoder = getHandle(encoderHandle);
            const caCount = readU32(descPtr + OFF.RPDesc2.ca_count);
            const caPtr = readU32(descPtr + OFF.RPDesc2.ca);
            const colorAttachments = [];
            for (let i = 0; i < caCount; i++) {
                const aPtr = caPtr + i * OFF.RPColorAtt.SIZEOF;
                const viewHandle = readU32(aPtr + OFF.RPColorAtt.view);
                const loadOp = LOAD_OP[readU32(aPtr + OFF.RPColorAtt.load_op)];
                const storeOp = STORE_OP[readU32(aPtr + OFF.RPColorAtt.store_op)];
                // Color clear_value: 4 doubles at +24
                const clearBase = aPtr + OFF.RPColorAtt.clear;
                const clearValue = {
                    r: readF64(clearBase),
                    g: readF64(clearBase + 8),
                    b: readF64(clearBase + 16),
                    a: readF64(clearBase + 24),
                };
                colorAttachments.push({
                    view: getHandle(viewHandle),
                    loadOp,
                    storeOp,
                    clearValue,
                });
            }
            // Read depth/stencil attachment (pointer at +20)
            let depthStencilAttachment = undefined;
            const rpDsPtr = readU32(descPtr + OFF.RPDesc2.ds);
            if (rpDsPtr) {
                depthStencilAttachment = {
                    view: getHandle(readU32(rpDsPtr + OFF.RPDSAtt.view)),
                    depthLoadOp: LOAD_OP[readU32(rpDsPtr + OFF.RPDSAtt.depth_load_op)],
                    depthStoreOp: STORE_OP[readU32(rpDsPtr + OFF.RPDSAtt.depth_store_op)],
                    depthClearValue: dv().getFloat32(rpDsPtr + OFF.RPDSAtt.depth_clear_value, true),
                };
            }
            const rpDesc = { colorAttachments };
            if (depthStencilAttachment) rpDesc.depthStencilAttachment = depthStencilAttachment;
            return addHandle(encoder.beginRenderPass(rpDesc));
        },

        // (i32 pass, i32 pipeline) -> void
        wgpuRenderPassEncoderSetPipeline(passHandle, pipelineHandle) {
            getHandle(passHandle).setPipeline(getHandle(pipelineHandle));
        },

        // (i32 pass, i32 group_index, i32 group, i32 dynamic_count, i32 dynamic_offsets_ptr) -> void
        wgpuRenderPassEncoderSetBindGroup(passHandle, groupIndex, groupHandle, dynamicCount, dynamicOffsetsPtr) {
            const pass = getHandle(passHandle);
            const group = getHandle(groupHandle);
            if (dynamicCount > 0 && dynamicOffsetsPtr) {
                const offsets = [];
                for (let i = 0; i < dynamicCount; i++) {
                    offsets.push(readU32(dynamicOffsetsPtr + i * 4));
                }
                pass.setBindGroup(groupIndex, group, offsets);
            } else {
                pass.setBindGroup(groupIndex, group);
            }
        },

        // (i32 pass, i32 slot, i32 buffer, i64 offset, i64 size) -> void
        wgpuRenderPassEncoderSetVertexBuffer(passHandle, slot, bufferHandle, offset, size) {
            getHandle(passHandle).setVertexBuffer(slot, getHandle(bufferHandle), Number(offset), Number(size));
        },

        // (i32 pass, i32 buffer, i32 format, i64 offset, i64 size) -> void
        wgpuRenderPassEncoderSetIndexBuffer(passHandle, bufferHandle, format, offset, size) {
            getHandle(passHandle).setIndexBuffer(
                getHandle(bufferHandle),
                INDEX_FORMAT[format],
                Number(offset),
                Number(size)
            );
        },

        // (i32 pass, i32 indexCount, i32 instanceCount, i32 firstIndex, i32 baseVertex, i32 firstInstance) -> void
        wgpuRenderPassEncoderDrawIndexed(passHandle, indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
            getHandle(passHandle).drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
        },

        // (i32 pass) -> void
        wgpuRenderPassEncoderEnd(passHandle) {
            getHandle(passHandle).end();
        },

        // (i32 encoder, i32 desc_ptr) -> i32 command_buffer
        wgpuCommandEncoderFinish(encoderHandle, descPtr) {
            return addHandle(getHandle(encoderHandle).finish());
        },

        // (i32 queue, i32 count, i32 commands_ptr) -> void
        wgpuQueueSubmit(queueHandle, count, commandsPtr) {
            const queue = getHandle(queueHandle);
            const cmds = [];
            for (let i = 0; i < count; i++) {
                cmds.push(getHandle(readU32(commandsPtr + i * 4)));
            }
            queue.submit(cmds);
        },

        // (i32 surface) -> i32 status
        wgpuSurfacePresent(surfaceHandle) {
            // Browser auto-presents, no-op. Return STATUS_SUCCESS = 1
            return 1;
        },
    };

    return {
        env,
        setMemory(mem) { wasmMemory = mem; },
        setPreInitialized(adapter, device, queue, canvasContext, canvasFormat) {
            preAdapter = adapter;
            preDevice = device;
            preQueue = queue;
            preCanvasContext = canvasContext;
            preCanvasFormat = canvasFormat;
        },
    };
}
