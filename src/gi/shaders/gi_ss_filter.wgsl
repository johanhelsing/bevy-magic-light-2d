#import bevy_magic_light_2d::gi_types::{LightPassParams, ProbeDataBuffer}
#import bevy_magic_light_2d::gi_math
#import bevy_magic_light_2d::gi_camera::{CameraParams, screen_to_world, screen_to_ndc, world_to_sdf_uv, bilinear_sample_r}
#import bevy_magic_light_2d::gi_halton
#import bevy_magic_light_2d::gi_attenuation
#import bevy_magic_light_2d::gi_raymarch::raymarch_primary

@group(0) @binding(0) var<uniform> camera_params:     CameraParams;
@group(0) @binding(1) var<uniform> cfg:               LightPassParams;
@group(0) @binding(2) var<storage> probes:            ProbeDataBuffer;
@group(0) @binding(3) var          sdf_in:            texture_2d<f32>;
@group(0) @binding(4) var          sdf_in_sampler:    sampler;
@group(0) @binding(5) var          ss_blend_in:       texture_storage_2d<rgba32float, read>;
@group(0) @binding(6) var          ss_filter_out:     texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var          ss_pose_out:      texture_storage_2d<rg32float, write>;

// Spatial Gaussian: standard bilateral, sigma = probe_size pixels.
// Adjacent probes are probe_size pixels apart → weight ≈ 0.61.
fn gauss_spatial(d_pixels: f32, probe_size: f32) -> f32 {
    let sigma = probe_size;
    return exp(-d_pixels * d_pixels / (2.0 * sigma * sigma));
}

// Range Gaussian: standard bilateral, sigma = 0.5 irradiance units.
// Preserves shadow edges while smoothing within uniform-lit regions.
fn gauss_range(irradiance_diff: f32) -> f32 {
    let sigma = 0.5;
    return exp(-irradiance_diff * irradiance_diff / (2.0 * sigma * sigma));
}


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let screen_pose        = vec2<i32>(invocation_id.xy);
    let sample_world_pose  = screen_to_world(
        screen_pose,
        camera_params.screen_size,
        camera_params.inverse_view_proj,
        camera_params.screen_size_inv,
    );

    let base_probe_grid_pose   = screen_pose / cfg.probe_size;
    // Reference irradiance: the blended probe that contains this pixel.
    // ss_blend_in is probe-grid-sized, so index with grid coords (not screen coords).
    let base_probe_sample      = textureLoad(ss_blend_in, base_probe_grid_pose).xyz;

    let kernel_hl = i32(cfg.smooth_kernel_size_w);
    let kernel_hr = i32(cfg.smooth_kernel_size_h);

    var total_w = 0.0;
    var total_q = vec3<f32>(0.0);
    var total_samples = 0;

    for (var i = -kernel_hl; i <= kernel_hl; i++) {
        for (var j = -kernel_hr; j <= kernel_hr; j++) {

            let offset = vec2<i32>(i, j);

            let p_grid_pose   = base_probe_grid_pose + offset;
            // Use probe tile center so distance is symmetric within the tile.
            let p_screen_pose = (base_probe_grid_pose + offset) * cfg.probe_size + cfg.probe_size / 2;

            // Discard offscreen;
            let p_ndc = screen_to_ndc(p_screen_pose, camera_params.screen_size, camera_params.screen_size_inv);
            if any(p_ndc < vec2<f32>(-1.0)) || any(p_ndc > vec2<f32>(1.0)) {
                continue;
            }

            let p_world_pose = screen_to_world(
                p_screen_pose,
                camera_params.screen_size,
                camera_params.inverse_view_proj,
                camera_params.screen_size_inv,
            );

            let p_sample = textureLoad(ss_blend_in, p_grid_pose).xyz;

            // Skip occlusion check if both points are inside an occluder.
            let sample_sdf = bilinear_sample_r(sdf_in, sdf_in_sampler,
                world_to_sdf_uv(sample_world_pose, camera_params.view_proj, camera_params.inv_sdf_scale));
            let probe_sdf = bilinear_sample_r(sdf_in, sdf_in_sampler,
                world_to_sdf_uv(p_world_pose, camera_params.view_proj, camera_params.inv_sdf_scale));
            let sdf_eps = camera_params.pixel_world_size.x * 2.0;
            let both_inside = sample_sdf <= sdf_eps && probe_sdf <= sdf_eps;
            if !both_inside && raymarch_primary(sample_world_pose, p_world_pose,
                8,
                sdf_in,
                sdf_in_sampler,
                camera_params,
                0.0).success <= 0 {
                continue;
            }

            let d = distance(p_world_pose, sample_world_pose);
            let x = distance(p_sample, base_probe_sample);
            let pws = camera_params.pixel_world_size.x;
            let probe_size_f = f32(cfg.probe_size);
            let g = gauss_spatial(d / pws, probe_size_f) * gauss_range(x);

            total_q += p_sample * g;
            total_w += g;
        }
    }

    var irradiance = vec3<f32>(0.0);
    if (total_w > 0.0) {
        irradiance = total_q / total_w;
    }

    let sdf_uv = world_to_sdf_uv(sample_world_pose, camera_params.view_proj, camera_params.inv_sdf_scale);

    textureStore(ss_filter_out, screen_pose, vec4<f32>(irradiance.xyz, 1.0));
    textureStore(ss_pose_out, screen_pose, vec4<f32>(sdf_uv, 0.0,  0.0));
}
