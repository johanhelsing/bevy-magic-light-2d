#import bevy_magic_light_2d::gi_types::{LightPassParams, ProbeDataBuffer}
#import bevy_magic_light_2d::gi_math
#import bevy_magic_light_2d::gi_camera::{CameraParams, screen_to_world, screen_to_ndc, world_to_sdf_uv, bilinear_sample_r}
#import bevy_magic_light_2d::gi_halton
#import bevy_magic_light_2d::gi_attenuation
#import bevy_magic_light_2d::gi_raymarch::{raymarch_primary, min_march_sdf}

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


// The blended probe field at a screen position, interpolated between the
// four probes whose cell centres surround it.
fn blend_bilinear(screen_pose: vec2<i32>) -> vec3<f32> {
    let p = (vec2<f32>(screen_pose) + 0.5) / f32(cfg.probe_size) - 0.5;
    let p0 = vec2<i32>(floor(p));
    let f = p - vec2<f32>(p0);
    let dims = vec2<i32>(textureDimensions(ss_blend_in)) - vec2<i32>(1);
    let a = clamp(p0, vec2<i32>(0), dims);
    let b = clamp(p0 + vec2<i32>(1, 0), vec2<i32>(0), dims);
    let c = clamp(p0 + vec2<i32>(0, 1), vec2<i32>(0), dims);
    let d = clamp(p0 + vec2<i32>(1, 1), vec2<i32>(0), dims);
    let top = mix(textureLoad(ss_blend_in, a).xyz, textureLoad(ss_blend_in, b).xyz, f.x);
    let bot = mix(textureLoad(ss_blend_in, c).xyz, textureLoad(ss_blend_in, d).xyz, f.x);
    return mix(top, bot, f.y);
}

// What the filter knows about the pixel it writes.
struct Pixel {
    world_pose:   vec2<f32>,
    // 1 outside every occluder, 0 inside, smooth across the boundary.
    outside:      f32,
    // Where its line-of-sight marches start.
    march_origin: vec2<f32>,
}

fn boundary_eps() -> f32 {
    return camera_params.pixel_world_size.x * f32(cfg.probe_size);
}

// Inside/outside classification of a point, from the SDF.
fn outside_at(world_pose: vec2<f32>) -> f32 {
    let sdf = bilinear_sample_r(sdf_in, sdf_in_sampler,
        world_to_sdf_uv(world_pose, camera_params.view_proj, camera_params.inv_sdf_scale));
    let eps = boundary_eps();
    return smoothstep(-eps, eps, sdf);
}

fn screen_pose_to_world(screen_pose: vec2<i32>) -> vec2<f32> {
    return screen_to_world(
        screen_pose,
        camera_params.screen_size,
        camera_params.inverse_view_proj,
        camera_params.screen_size_inv,
    );
}

fn pixel_at(screen_pose: vec2<i32>) -> Pixel {
    let sample_world_pose = screen_pose_to_world(screen_pose);

    // Inside/outside classification at the pixel itself. The SDF is the one
    // smooth signal we have at pixel resolution; classifying at the cell
    // centre instead quantized a wall's lit face (the strip between a piece
    // and its shrunk occluder, a few pixels) to whole 4-pixel cells, so a
    // face lit up or not by where it fell on the cell grid. The smoothstep
    // keeps a moving occluder's edge from popping.
    let sample_sdf = bilinear_sample_r(sdf_in, sdf_in_sampler,
        world_to_sdf_uv(sample_world_pose, camera_params.view_proj, camera_params.inv_sdf_scale));
    let eps = boundary_eps();
    let sample_outside = smoothstep(-eps, eps, sample_sdf);
    // The march fails at once from a point nearer the boundary than its
    // own minimum distance, which every pixel on the lit face is. Such a
    // pixel marches from itself pushed out along the SDF gradient to that
    // distance, so the face is classified at pixel resolution and still
    // sees only the probes it has line of sight to.
    let min_sdf = min_march_sdf(camera_params);
    var march_origin = sample_world_pose;
    if sample_sdf < min_sdf {
        let h = camera_params.pixel_world_size.x;
        let sdf_at = sample_world_pose + vec2<f32>(h, 0.0);
        let sdf_up = sample_world_pose + vec2<f32>(0.0, h);
        let dx = bilinear_sample_r(sdf_in, sdf_in_sampler,
            world_to_sdf_uv(sdf_at, camera_params.view_proj, camera_params.inv_sdf_scale)) - sample_sdf;
        let dy = bilinear_sample_r(sdf_in, sdf_in_sampler,
            world_to_sdf_uv(sdf_up, camera_params.view_proj, camera_params.inv_sdf_scale)) - sample_sdf;
        let grad = vec2<f32>(dx, dy);
        let len = length(grad);
        if len > 0.0 {
            march_origin += grad / len * (min_sdf - sample_sdf + h * 0.5);
        }
    }

    return Pixel(sample_world_pose, sample_outside, march_origin);
}

// The bilateral weight of one tap for the pixel, 0 when the pixel cannot
// see it. `d_pixels` is the screen distance the spatial term reads.
fn tap_weight(
    px:          Pixel,
    reference:   vec3<f32>,
    tap_world:   vec2<f32>,
    tap_outside: f32,
    tap_value:   vec3<f32>,
    d_pixels:    f32,
) -> f32 {
    // Penalise cross-boundary pairs: falls to 0 when sample and tap are
    // on opposite sides, stays near 1 when they are on the same side.
    let cross_w = 1.0 - abs(px.outside - tap_outside);
    if cross_w <= 0.0 { return 0.0; }

    // Only raymarch when both points are clearly outside an occluder.
    let do_occlusion = px.outside * tap_outside;
    if do_occlusion > 0.5 && raymarch_primary(px.march_origin, tap_world,
        8,
        sdf_in,
        sdf_in_sampler,
        camera_params,
        0.0).success <= 0 {
        return 0.0;
    }

    let x = distance(tap_value, reference);
    return gauss_spatial(d_pixels, f32(cfg.probe_size)) * gauss_range(x) * cross_w;
}

// The probes within `half` cells of the pixel's cell, weighted.
fn gather_probes(screen_pose: vec2<i32>, px: Pixel, half: vec2<i32>) -> vec3<f32> {
    let base_probe_grid_pose = screen_pose / cfg.probe_size;

    // Reference irradiance for the range term: the blended probes around
    // this pixel, bilinearly interpolated at the pixel's position. One probe
    // per cell as the reference stepped the output at every cell edge
    // wherever neighboring probes differ, which is every occluder boundary.
    let base_probe_sample = blend_bilinear(screen_pose);

    var total_w = 0.0;
    var total_q = vec3<f32>(0.0);

    for (var i = -half.x; i <= half.x; i++) {
        for (var j = -half.y; j <= half.y; j++) {

            let offset = vec2<i32>(i, j);

            let p_grid_pose   = base_probe_grid_pose + offset;
            // Use probe tile center so distance is symmetric within the tile.
            let p_screen_pose = (base_probe_grid_pose + offset) * cfg.probe_size + cfg.probe_size / 2;

            // Discard offscreen;
            let p_ndc = screen_to_ndc(p_screen_pose, camera_params.screen_size, camera_params.screen_size_inv);
            if any(p_ndc < vec2<f32>(-1.0)) || any(p_ndc > vec2<f32>(1.0)) {
                continue;
            }

            let p_world_pose = screen_pose_to_world(p_screen_pose);
            let p_sample = textureLoad(ss_blend_in, p_grid_pose).xyz;
            let d = distance(p_world_pose, px.world_pose) / camera_params.pixel_world_size.x;
            let g = tap_weight(px, base_probe_sample, p_world_pose, outside_at(p_world_pose), p_sample, d);

            total_q += p_sample * g;
            total_w += g;
        }
    }

    if total_w > 0.0 {
        return total_q / total_w;
    }
    return vec3<f32>(0.0);
}

fn store_pose(screen_pose: vec2<i32>, px: Pixel) {
    let sdf_uv = world_to_sdf_uv(px.world_pose, camera_params.view_proj, camera_params.inv_sdf_scale);
    textureStore(ss_pose_out, screen_pose, vec4<f32>(sdf_uv, 0.0,  0.0));
}

// The full 2D kernel: every probe within `smooth_kernel_size` cells.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let screen_pose = vec2<i32>(invocation_id.xy);
    let px = pixel_at(screen_pose);
    let half = vec2<i32>(i32(cfg.smooth_kernel_size_w), i32(cfg.smooth_kernel_size_h));
    let irradiance = gather_probes(screen_pose, px, half);
    textureStore(ss_filter_out, screen_pose, vec4<f32>(irradiance, 1.0));
    store_pose(screen_pose, px);
}

// The separable kernel, first half: the probes in the pixel's own row of
// cells, at full resolution into an intermediate target. The spatial term
// is the full distance, so it carries a factor for the pixel's offset from
// its row, common to every tap, which the normalization cancels.
@compute @workgroup_size(8, 8, 1)
fn main_h(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let screen_pose = vec2<i32>(invocation_id.xy);
    let px = pixel_at(screen_pose);
    let half = vec2<i32>(i32(cfg.smooth_kernel_size_w), 0);
    let irradiance = gather_probes(screen_pose, px, half);
    textureStore(ss_filter_out, screen_pose, vec4<f32>(irradiance, 1.0));
}

// The separable kernel, second half: `ss_blend_in` holds the first half's
// output. Taps the pixels one cell apart in the pixel's column, each the
// smoothed row of cells it sits in, weighted by the pixel's distance to
// that row. Light reaches the pixel around one corner: from a probe to
// the tap along the row, then from the tap to the pixel.
@compute @workgroup_size(8, 8, 1)
fn main_v(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let screen_pose = vec2<i32>(invocation_id.xy);
    let px = pixel_at(screen_pose);
    let half = i32(cfg.smooth_kernel_size_h);
    let cell_y = screen_pose.y / cfg.probe_size;
    let screen_h = i32(camera_params.screen_size.y);
    let reference = textureLoad(ss_blend_in, screen_pose).xyz;

    // The pixel's own row needs no march.
    let d0 = f32(abs(cell_y * cfg.probe_size + cfg.probe_size / 2 - screen_pose.y));
    var total_w = gauss_spatial(d0, f32(cfg.probe_size));
    var total_q = reference * total_w;

    for (var j = -half; j <= half; j++) {
        if j == 0 { continue; }
        let tap = screen_pose + vec2<i32>(0, j * cfg.probe_size);
        if tap.y < 0 || tap.y >= screen_h { continue; }

        let tap_world = screen_pose_to_world(tap);
        let tap_value = textureLoad(ss_blend_in, tap).xyz;
        let row_center = (cell_y + j) * cfg.probe_size + cfg.probe_size / 2;
        let d = f32(abs(row_center - screen_pose.y));
        let g = tap_weight(px, reference, tap_world, outside_at(tap_world), tap_value, d);

        total_q += tap_value * g;
        total_w += g;
    }

    textureStore(ss_filter_out, screen_pose, vec4<f32>(total_q / total_w, 1.0));
    store_pose(screen_pose, px);
}
