module webgpu;

import c3w;
import std::io;
import std::math;

// ----------------------------------------------------------------
// Inline WGSL shader
// ----------------------------------------------------------------
const char* WGSL_SOURCE = `
struct Uniforms {
    mvp: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexIn {
    @location(0) pos:   vec3<f32>,
    @location(1) color: vec4<f32>,
}
struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       color:    vec4<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = uniforms.mvp * vec4<f32>(in.pos, 1.0);
    out.color    = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
`;

// ----------------------------------------------------------------
// Cube geometry  (6 faces × 4 verts = 24 verts, 36 indices)
// Each vertex: float[3] pos + float[4] color
// ----------------------------------------------------------------
const float[7 * 24] VERTICES = {
    // Front face (red)
    -1, -1,  1,   1,0,0,1,
     1, -1,  1,   1,0,0,1,
     1,  1,  1,   1,0,0,1,
    -1,  1,  1,   1,0,0,1,
    // Back face (green)
     1, -1, -1,   0,1,0,1,
    -1, -1, -1,   0,1,0,1,
    -1,  1, -1,   0,1,0,1,
     1,  1, -1,   0,1,0,1,
    // Left face (blue)
    -1, -1, -1,   0,0,1,1,
    -1, -1,  1,   0,0,1,1,
    -1,  1,  1,   0,0,1,1,
    -1,  1, -1,   0,0,1,1,
    // Right face (yellow)
     1, -1,  1,   1,1,0,1,
     1, -1, -1,   1,1,0,1,
     1,  1, -1,   1,1,0,1,
     1,  1,  1,   1,1,0,1,
    // Top face (cyan)
    -1,  1,  1,   0,1,1,1,
     1,  1,  1,   0,1,1,1,
     1,  1, -1,   0,1,1,1,
    -1,  1, -1,   0,1,1,1,
    // Bottom face (magenta)
    -1, -1, -1,   1,0,1,1,
     1, -1, -1,   1,0,1,1,
     1, -1,  1,   1,0,1,1,
    -1, -1,  1,   1,0,1,1,
};

const uint[36] INDICES = {
     0,  1,  2,   0,  2,  3,   // front
     4,  5,  6,   4,  6,  7,   // back
     8,  9, 10,   8, 10, 11,   // left
    12, 13, 14,  12, 14, 15,   // right
    16, 17, 18,  16, 18, 19,   // top
    20, 21, 22,  20, 22, 23,   // bottom
};

// Matrix type alias for convenience (stdlib row-major Matrix4x4).
// Upload to GPU with .transpose() since WGSL mat4x4 is column-major.
alias Mat4 = math::Matrix4f;

// ----------------------------------------------------------------
// Adapter / Device callback state
// ----------------------------------------------------------------

struct AdapterResult { Adapter adapter; bool done; }
struct DeviceResult  { Device  device;  bool done; }

fn void on_adapter(RequestAdapterStatus status, Adapter adapter, StringView msg, void* ud1, void* ud2) {
    AdapterResult* r = (AdapterResult*)ud1;
    if (status == REQUEST_ADAPTER_STATUS_SUCCESS) {
        r.adapter = adapter;
    }
    r.done = true;
}

fn void on_device(RequestDeviceStatus status, Device device, StringView msg, void* ud1, void* ud2) {
    DeviceResult* r = (DeviceResult*)ud1;
    if (status == REQUEST_DEVICE_STATUS_SUCCESS) {
        r.device = device;
    }
    r.done = true;
}

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------

fn void main() {
    // --- Window ---
    c3w::Params params = { .x = 100, .y = 100, .width = 800, .height = 600 };
    c3w::Window window = c3w::new(params, "WebGPU Cube")!!;
    defer window.free();

    // --- WebGPU Instance ---
    InstanceDescriptor inst_desc;
    Instance instance = createInstance(&inst_desc);

    // --- Surface ---
    Surface surface;
    $if env::LINUX:
        switch (window.src.backend) {
            case c3w::BackendType.WAYLAND:
                WaylandWindow* wl = (WaylandWindow*)window.src.backend_data;
                SurfaceSourceWaylandSurface wl_src = {
                    .chain_next = null,
                    .s_type     = S_TYPE_SURFACE_SOURCE_WAYLAND_SURFACE,
                    .display    = wl.display,
                    .surface    = wl.surface,
                };
                SurfaceDescriptor surf_desc = {
                    .next_in_chain = &wl_src,
                    .label         = sv_empty(),
                };
                surface = instanceCreateSurface(instance, &surf_desc);
            case c3w::BackendType.X11:
                X11Window* x11 = (X11Window*)window.src.backend_data;
                SurfaceSourceXCBWindow xcb_src = {
                    .chain_next  = null,
                    .s_type      = S_TYPE_SURFACE_SOURCE_XCB_WINDOW,
                    .connection  = x11.connection,
                    .window      = (uint)(uptr)x11.window,
                };
                SurfaceDescriptor surf_desc = {
                    .next_in_chain = &xcb_src,
                    .label         = sv_empty(),
                };
                surface = instanceCreateSurface(instance, &surf_desc);
            default:
                io::printfn("Unsupported display backend");
                return;
        }
    $endif

    // --- Request Adapter ---
    AdapterResult adapter_result;
    RequestAdapterOptions adapter_opts = {
        .compatible_surface  = surface,
        .power_preference    = POWER_PREFERENCE_HIGH_PERFORMANCE,
        .feature_level       = FEATURE_LEVEL_CORE,
    };
    RequestAdapterCallbackInfo adapter_cb = {
        .mode      = CALLBACK_MODE_ALLOW_SPONTANEOUS,
        .callback  = &on_adapter,
        .userdata1 = &adapter_result,
    };
    instanceRequestAdapter(instance, &adapter_opts, adapter_cb);
    while (!adapter_result.done) { instanceProcessEvents(instance); }
    Adapter adapter = adapter_result.adapter;

    // --- Request Device ---
    DeviceResult device_result;
    DeviceDescriptor dev_desc;
    RequestDeviceCallbackInfo device_cb = {
        .mode      = CALLBACK_MODE_ALLOW_SPONTANEOUS,
        .callback  = &on_device,
        .userdata1 = &device_result,
    };
    adapterRequestDevice(adapter, &dev_desc, device_cb);
    while (!device_result.done) { instanceProcessEvents(instance); }
    Device device = device_result.device;

    Queue queue = deviceGetQueue(device);

    // --- Surface capabilities & configure ---
    SurfaceCapabilities caps;
    surfaceGetCapabilities(surface, adapter, &caps);
    TextureFormat surf_format = caps.formats_count > 0 ? caps.formats[0] : TEXTURE_FORMAT_BGRA8_UNORM;

    SurfaceConfiguration surf_config = {
        .device       = device,
        .format       = surf_format,
        .usage        = TEXTURE_USAGE_RENDER_ATTACHMENT,
        .width        = (uint)window.width,
        .height       = (uint)window.height,
        .alpha_mode   = COMPOSITE_ALPHA_MODE_AUTO,
        .present_mode = PRESENT_MODE_FIFO,
    };
    surfaceConfigure(surface, &surf_config);

    // --- Vertex buffer ---
    GpuBuffer vb = gpu_buffer(device, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST, VERTICES.len * float.sizeof);
    vb.upload(queue, &VERTICES, VERTICES.len * float.sizeof);

    // --- Index buffer ---
    GpuBuffer ib = gpu_buffer(device, BUFFER_USAGE_INDEX | BUFFER_USAGE_COPY_DST, INDICES.len * uint.sizeof);
    ib.upload(queue, &INDICES, INDICES.len * uint.sizeof);

    // --- Uniform buffer (64 bytes = one mat4) ---
    GpuBuffer ub = gpu_buffer(device, BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST, 64);

    // --- Bind group layout (one uniform buffer at binding 0) ---
    BufferBindingLayout buf_binding = {
        .type              = BUFFER_BINDING_TYPE_UNIFORM,
        .has_dynamic_offset = 0,
        .min_binding_size  = 64,
    };
    BindGroupLayoutEntry bgl_entry = {
        .binding    = 0,
        .visibility = SHADER_STAGE_VERTEX,
        .buffer     = buf_binding,
    };
    BindGroupLayoutDescriptor bgl_desc = {
        .label         = sv("Cube BGL"),
        .entries_count = 1,
        .entries       = &bgl_entry,
    };
    BindGroupLayout bgl = deviceCreateBindGroupLayout(device, &bgl_desc);

    // --- Pipeline layout ---
    PipelineLayoutDescriptor pl_desc = {
        .label                  = sv("Cube PL"),
        .bind_group_layouts_count = 1,
        .bind_group_layouts     = &bgl,
    };
    PipelineLayout pipeline_layout = deviceCreatePipelineLayout(device, &pl_desc);

    // --- Bind group ---
    BindGroupEntry bg_entry = {
        .binding = 0,
        .buffer  = ub.handle,
        .offset  = 0,
        .size    = 64,
    };
    BindGroupDescriptor bg_desc = {
        .label         = sv("Cube BG"),
        .layout        = bgl,
        .entries_count = 1,
        .entries       = &bg_entry,
    };
    BindGroup bind_group = deviceCreateBindGroup(device, &bg_desc);

    // --- Shader module ---
    ShaderSourceWGSL wgsl_src = {
        .chain_next = null,
        .s_type     = S_TYPE_SHADER_SOURCE_WGSL,
        .code       = sv(WGSL_SOURCE),
    };
    ShaderModuleDescriptor sm_desc = {
        .next_in_chain = &wgsl_src,
        .label         = sv("Cube Shader"),
    };
    ShaderModule shader = deviceCreateShaderModule(device, &sm_desc);

    // --- Vertex buffer layout ---
    VertexAttribute[2] vert_attrs = {
        // position: float32x3 at offset 0
        { .format = VERTEX_FORMAT_FLOAT32X3, .offset = 0, .shader_location = 0 },
        // color: float32x4 at offset 12
        { .format = VERTEX_FORMAT_FLOAT32X4, .offset = 12, .shader_location = 1 },
    };
    VertexBufferLayout vbl = {
        .step_mode        = VERTEX_STEP_MODE_VERTEX,
        .array_stride     = 28,   // 3*4 + 4*4 bytes
        .attributes_count = 2,
        .attributes       = &vert_attrs,
    };

    // --- Render pipeline ---
    ColorTargetState color_target = {
        .format     = surf_format,
        .write_mask = COLOR_WRITE_MASK_ALL,
    };
    FragmentState frag_state = {
        .shader_module  = shader,
        .entry_point    = sv("fs_main"),
        .targets_count  = 1,
        .targets        = &color_target,
    };
    VertexState vert_state = {
        .shader_module  = shader,
        .entry_point    = sv("vs_main"),
        .buffers_count  = 1,
        .buffers        = &vbl,
    };
    PrimitiveState prim_state = {
        .topology   = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .front_face = FRONT_FACE_CCW,
        .cull_mode  = CULL_MODE_BACK,
    };
    MultisampleState ms_state = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
    };
    RenderPipelineDescriptor rp_desc = {
        .label       = sv("Cube Pipeline"),
        .layout      = pipeline_layout,
        .vertex      = vert_state,
        .primitive   = prim_state,
        .multisample = ms_state,
        .fragment    = &frag_state,
    };
    RenderPipeline pipeline = deviceCreateRenderPipeline(device, &rp_desc);

    // --- Projection & view ---
    // Stdlib matrices are row-major; transpose before uploading since WGSL is column-major.
    float aspect = (float)window.width / (float)window.height;
    Mat4 proj = matrix4f_perspective((float)math::PI / 4.0f, aspect, 0.1f, 100.0f);
    Mat4 view = MATRIX4F_IDENTITY.translate({0.0f, 0.0f, -4.0f});

    float angle = 0.0f;

    // ---- Render loop ----
    bool running = true;
    while (running) {
        c3w::EventMap events = window.getEvent();

        if (events.is_pressed(c3w::EventKey.ESCAPE) ||
            events.is_pressed(c3w::EventKey.Q)) {
            running = false;
            break;
        }

        // Rotate cube, build MVP, transpose for WGSL column-major upload
        angle += 0.01f;
        Mat4 model   = MATRIX4F_IDENTITY.rotate_y(angle);
        Mat4 mvp     = proj * view * model;
        Mat4 mvp_t   = mvp.transpose();
        ub.upload(queue, &mvp_t, 64);

        // Get surface texture
        SurfaceTexture surf_tex;
        surfaceGetCurrentTexture(surface, &surf_tex);
        if (surf_tex.status != SURFACE_GET_CURRENT_TEXTURE_STATUS_SUCCESS_OPTIMAL &&
            surf_tex.status != SURFACE_GET_CURRENT_TEXTURE_STATUS_SUCCESS_SUBOPTIMAL) {
            continue;
        }

        TextureView frame_view = textureCreateView(surf_tex.texture, null);

        // Encode commands
        CommandEncoderDescriptor enc_desc;
        CommandEncoder encoder = deviceCreateCommandEncoder(device, &enc_desc);

        RenderPassColorAttachment color_att = {
            .view        = frame_view,
            .depth_slice = WGPU_DEPTH_SLICE_UNDEFINED,
            .load_op     = LOAD_OP_CLEAR,
            .store_op    = STORE_OP_STORE,
            .clear_value = { 0.1, 0.1, 0.15, 1.0 },
        };
        RenderPassDescriptor rp_desc2 = {
            .label                   = sv("Frame"),
            .color_attachments_count = 1,
            .color_attachments       = &color_att,
        };
        RenderPassEncoder pass = commandEncoderBeginRenderPass(encoder, &rp_desc2);

        renderPassEncoderSetPipeline(pass, pipeline);
        renderPassEncoderSetBindGroup(pass, 0, bind_group, 0, null);
        renderPassEncoderSetVertexBuffer(pass, 0, vb.handle, 0, vb.size);
        renderPassEncoderSetIndexBuffer(pass, ib.handle, INDEX_FORMAT_UINT32, 0, ib.size);
        renderPassEncoderDrawIndexed(pass, 36, 1, 0, 0, 0);
        renderPassEncoderEnd(pass);

        CommandBufferDescriptor cb_desc;
        CommandBuffer cmd = commandEncoderFinish(encoder, &cb_desc);
        queueSubmit(queue, 1, &cmd);
        surfacePresent(surface);
    }

    // Cleanup
    vb.destroy();
    ib.destroy();
    ub.destroy();
    surfaceUnconfigure(surface);
}
