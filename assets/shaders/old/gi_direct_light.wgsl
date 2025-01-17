#import bevy_2d_gi_experiment::gi_types
#import bevy_2d_gi_experiment::gi_camera
#import bevy_2d_gi_experiment::gi_halton
#import bevy_2d_gi_experiment::gi_attenuation

@group(0) @binding(0) var<uniform> camera_params:         CameraParams;
@group(0) @binding(1) var<storage> probes:             ProbeDataBuffer;
@group(0) @binding(2) var<uniform> state:              GiState;
@group(0) @binding(3) var          ss_grid_texture_in:    texture_storage_2d<rgba16float, read>;
@group(0) @binding(4) var          sdf_texture_in:        texture_storage_2d<r16float,    read>;
@group(0) @binding(5) var          target_out:            texture_storage_2d<rgba32float, read_write>;


fn distance_squared(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let c = a - b;
    return dot(c, c);
}

fn get_sdf_screen(screen_pose: vec2<i32>) -> f32 {
    return textureLoad(sdf_texture_in, screen_pose).r;
}

fn raymarch_occlusion(
    ray_origin:    vec2<f32>,
    light_pose:    vec2<f32>,
) -> f32 {

    let max_steps      = 6;
    let ray_direction  = normalize(light_pose - ray_origin);
    let stop_at        = distance_squared(ray_origin, light_pose);

    var ray_progress   = 0.0;
    for (var i: i32 = 0; i < max_steps; i++) {

        if (ray_progress * ray_progress >= stop_at) {
            return 1.0;
        }

        let h          = ray_origin + ray_progress * ray_direction;
        let scene_dist = get_sdf_screen(world_to_screen(h, camera_params.screen_size, camera_params.view_proj));

        if (scene_dist <= 0.0) {
            return 0.0;
        }

        ray_progress += scene_dist;
    }

    return 0.0;
}

struct ProbeVal {
    val:       vec3<f32>,
    pose:      vec2<f32>,
}

fn read_probe(
    probe_tile_origin: vec2<i32>,
    probe_tile_pose:   vec2<i32>,
    probe_offset:      vec2<i32>,
    motion_offset:     vec2<f32>,
    tile_size:         vec2<i32>,
    probe_size_f32:    f32) -> ProbeVal {

    let clamped_offset = clamp(probe_tile_pose + probe_offset, vec2<i32>(0), tile_size - vec2<i32>(1));

    // Get position
    let probe_screen_pose = clamped_offset * state.ss_probe_size;
    let probe_atlas_pose  = probe_tile_origin + clamped_offset;

    //
    let data        = textureLoad(ss_grid_texture_in, probe_atlas_pose);
    var val         = data.xyz;

    let halton_offset  = unpack2x16float(bitcast<u32>(data.w)) * probe_size_f32 * 1.0;
    let probe_pose     = screen_to_world(
        probe_screen_pose,
        camera_params.screen_size,
        camera_params.inverse_view_proj) + halton_offset - motion_offset;

    return ProbeVal(
        val,
        probe_pose,
    );
}

struct SampleResult {
    val:    vec3<f32>,
    weight: f32,
}

fn get_probe_tile_origin(
    probe_id:       i32,
    rows:           i32,
    cols:           i32,
    probe_size:     i32) -> vec2<i32> {

    return vec2<i32>(
        cols,
        rows,
    ) * vec2<i32>(probe_id % probe_size, probe_id / probe_size);
}

fn gauss(x: f32) -> f32 {
    let a = 4.0;
    let b = 0.2;
    let c = 0.05;

    let d = 1.0 / (2.0 * c * c);

    return a * exp(- (x - b) * (x - b) / d);
}

fn estimate_probes_at(
    sample_pose:         vec2<f32>,
    screen_pose:         vec2<i32>,
    probe_id:            i32,
    probe_camera_motion: vec2<f32>,
    tile_size:           vec2<i32>,
    probe_size_f32:      f32) -> SampleResult {

    // Reproject sample world pose to previous frame world pose.
    let reproj_sample_pose     = sample_pose + probe_camera_motion;
    let reproj_ndc             = world_to_ndc(reproj_sample_pose, camera_params.view_proj);

    // Probe pose in the screen.
    let reproj_screen_pose     = ndc_to_screen(reproj_ndc.xy, camera_params.screen_size);

    // Probe pose in tile.
    let reproj_tile_probe_pose = reproj_screen_pose / state.ss_probe_size;

    // Get origin position of the probe tile in the atlas.
    let curr_probe_origin      = get_probe_tile_origin(
        probe_id,
        state.ss_atlas_rows,
        state.ss_atlas_cols,
        state.ss_probe_size,
    );

    // Use current "central" tile probe as a base.
    let base_offset = vec2<i32>(0, 0);
    let base_probe  = read_probe(
        curr_probe_origin,
        reproj_tile_probe_pose,
        base_offset,
        probe_camera_motion,
        tile_size,
        probe_size_f32);

    // Discard if offscreen.
    let base_ndc = world_to_ndc(base_probe.pose, camera_params.view_proj);
    if any(base_ndc < vec2<f32>(-1.0)) || any(base_ndc > vec2<f32>(1.0)) {
        return SampleResult(vec3<f32>(0.0), 0.0);
    }

    // Bilateral filter kernel size.
    // 4x4 kernel: [-1, 0, 1, 2] x [-1, 0, 1, 2]
    let kernel_hl   = 0;
    let kernel_hr   = 0;

    var total_q = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = -kernel_hl; i <= kernel_hr; i++) {
        for (var j = -kernel_hl; j <= kernel_hr; j++) {
    // for (var i = 0; i < 4; i ++) {
        // let offset = vec2<i32>(i / 2, i % 2);
        let offset = vec2<i32>(i, j);
        let p = read_probe(curr_probe_origin, reproj_tile_probe_pose, offset, probe_camera_motion, tile_size, probe_size_f32);
        let d = distance(p.pose, sample_pose);

        // Discard if probe is too far away.
        if d > probe_size_f32 * 1.33 {
            continue;
        }

        // Discard if offscreen.
        let p_ndc = world_to_ndc(p.pose, camera_params.view_proj);
        if any(p_ndc < vec2<f32>(-1.0)) || any(p_ndc > vec2<f32>(1.0)) {
            continue;
        }

        // Discard occluded probes.
        if raymarch_occlusion(sample_pose, p.pose) <= 0.0 {
            continue;
        }

        // Compute bilateral filter with gauss function
        let x = distance(p.val, base_probe.val);
        let g = gauss(x) * gauss(d);

        total_q += p.val * g;
        total_w += g;

    // }
        }
    }

    return SampleResult(
        clamp(total_q, vec3<f32>(0.0), vec3<f32>(1e+4)),
        clamp(total_w, 0.0, 1e+4),
    );
}


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let screen_pose  = vec2<i32>(invocation_id.xy);
    let sample_pose  = screen_to_world(screen_pose, camera_params.screen_size, camera_params.inverse_view_proj);

    let reservoir_size     = 8;
    let curr_probe_id      = state.gi_frame_counter % reservoir_size;

    let camera_buffer_size = state.ss_probe_size * state.ss_probe_size;
    let camera_buffer_id   = state.gi_frame_counter;
    let curr_camera_pose   = probes.data[camera_buffer_id].pose;
    let probe_size_f32     = f32(state.ss_probe_size);

    let tile_size          = vec2<i32>(camera_params.screen_size / (f32(state.ss_probe_size) - 0.001));
    let min_irradiance     = vec3<f32>(0.0);
    let max_irradiance     = vec3<f32>(1e+4);
    var total_irradiance   = min_irradiance;
    var total_weight       = 0.0;

    // Sample radiance from previous frames.
    for (var i = 0; i < reservoir_size; i++) {

        // Get index of probe tile of previous frame.
        var probe_id = curr_probe_id - i;
        if (probe_id < 0) {
            probe_id = reservoir_size + probe_id;
        }

        // Get index of camera of previous frame.
        var probe_camera_buffer_id  = camera_buffer_id - i;
        if (probe_camera_buffer_id < 0) {
            probe_camera_buffer_id = camera_buffer_size + probe_camera_buffer_id;
        }

        // Compute position change.
        let probe_camera_pose   = probes.data[probe_camera_buffer_id].pose;
        let probe_camera_motion = curr_camera_pose - probe_camera_pose;

        // Get sample probe value.
        let r = estimate_probes_at(
            sample_pose,
            screen_pose,
            probe_id,
            probe_camera_motion,
            tile_size,
            probe_size_f32,
        );


        // If probe is active, accumulate irradiance and weight.
        if r.weight > 0.0 {
            total_irradiance += clamp(r.val, min_irradiance, max_irradiance);
            total_weight     += r.weight;
        }
    }

    // Shadow preserving temporal blending.
    // {
    //     let cur = total_irradiance;
    //     let old = textureLoad(target_out, screen_pose).xyz;
    //     let l1 = dot(cur, vec3<f32>(1.0 / 3.0));
    //     let l2 = dot(old, vec3<f32>(1.0 / 3.0));
    //     var a  = max(l1 - l2 - min(l1, l2), 0.0) / max(max(l1, l2), 1e-4);
    //         a  = clamp(a, 0.0, 0.95);
    //         a  = a * a;
    //     total_irradiance = mix(old, cur, vec3<f32>(a));
    // }

    // Normalize and clamp.
    total_irradiance = total_irradiance / total_weight;
    total_irradiance = clamp(total_irradiance, min_irradiance, max_irradiance);

    textureStore(target_out, screen_pose, vec4<f32>(total_irradiance, 1.0));

    // Uncomment to show probe atlas.
    // {
    //     let offset = vec2<i32>(160, 90) * vec2<i32>(
    //         curr_probe_id % state.ss_probe_size,
    //         curr_probe_id / state.ss_probe_size
    //     );
    //     textureStore(target_out, screen_pose, vec4<f32>(textureLoad(ss_grid_texture_in, screen_pose / 8 + offset).xyz, 1.0));
    // }
}