#import bevy_magic_light_2d::gi_types::{LightPassParams, ProbeDataBuffer, SkylightMaskBuffer, LightSourceBuffer}
#import bevy_magic_light_2d::gi_math
#import bevy_magic_light_2d::gi_camera::{CameraParams, world_to_sdf_uv, bilinear_sample_rgba, screen_to_world, world_to_ndc, ndc_to_screen, bilinear_sample_r, sdf_interior, probe_jitter_offset_world}
#import bevy_magic_light_2d::gi_attenuation::light_attenuation_r_two
#import bevy_magic_light_2d::gi_halton::hammersley2d
#import bevy_magic_light_2d::gi_raymarch::{raymarch_bounce, raymarch_primary, lit_from, interior_fade, exit_toward}

@group(0) @binding(0) var<uniform> camera_params:         CameraParams;
@group(0) @binding(1) var<uniform> cfg:                   LightPassParams;
@group(0) @binding(2) var<storage> probes:                ProbeDataBuffer;
@group(0) @binding(3) var<storage> skylight_masks_buffer: SkylightMaskBuffer;
@group(0) @binding(4) var<storage> lights_source_buffer:  LightSourceBuffer;
@group(0) @binding(5) var          sdf_in:                texture_2d<f32>;
@group(0) @binding(6) var          sdf_in_sampler:        sampler;
@group(0) @binding(7) var          ss_probe_out:          texture_storage_2d<rgba16float, write>;


// The probe's tile of the screen-space cache for this frame.
fn store_probe(tile_xy: vec2<i32>, frame_index: i32, halton_jitter: vec2<f32>, irradiance: vec3<f32>) {
    let atlas_row  = frame_index / cfg.probe_size;
    let atlas_col  = frame_index % cfg.probe_size;
    let out_atlas_tile_offset = vec2<i32>(
        cfg.probe_atlas_cols * atlas_col,
        cfg.probe_atlas_rows * atlas_row,
    );
    let out_atlas_tile_pose = out_atlas_tile_offset + tile_xy;
    let out_halton_jitter   = pack2x16float(halton_jitter);
    textureStore(ss_probe_out, out_atlas_tile_pose, vec4<f32>(irradiance, bitcast<f32>(out_halton_jitter)));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let tile_xy      = vec2<i32>(invocation_id.xy);

    // Screen-space position of the probe.
    let reservoir_size           = i32(cfg.reservoir_size);
    let probe_size_f32           = f32(cfg.probe_size);
    let frames_max               = cfg.probe_size * cfg.probe_size;
    let frame_index              = cfg.frame_counter % reservoir_size;
    let halton_jitter            = hammersley2d(frame_index, reservoir_size);
    let probe_tile_origin_screen = tile_xy * cfg.probe_size;

    // Get current frame.
    let probe_offset_world  = probe_jitter_offset_world(halton_jitter, probe_size_f32, camera_params.pixel_world_size);
    let probe_center_world_unbiased = screen_to_world(
                                              probe_tile_origin_screen,
                                              camera_params.screen_size,
                                              camera_params.inverse_view_proj,
                                              camera_params.screen_size_inv,
                                          );
    let probe_center_world  =  probe_center_world_unbiased + probe_offset_world;

    let probe_ndc    = world_to_ndc(probe_center_world, camera_params.view_proj);
    let probe_screen = ndc_to_screen(probe_ndc, camera_params.screen_size);
    var is_masked    = 1.0;

    // Check if the probe is masked from skylight.
    for (var i: i32 = 0; i < i32(skylight_masks_buffer.count); i++) {
        let mask = skylight_masks_buffer.data[i];
        if probe_center_world.x > mask.center.x - mask.h_extent.x &&
           probe_center_world.x < mask.center.x + mask.h_extent.x &&
           probe_center_world.y > mask.center.y - mask.h_extent.y &&
           probe_center_world.y < mask.center.y + mask.h_extent.y {
            is_masked = 0.0;
            break;
        }
    }

    let skylight = cfg.skylight_color * is_masked;
    var probe_irradiance = vec3<f32>(skylight);

    // A probe inside an occluder is lit by each light through the occluder
    // toward that light, faded by how far it crosses inside, as the occluder
    // says; one on the boundary band outside marches from itself pushed out,
    // so the face it is on is lit and the far side of an occluder is not lit
    // through it. The skylight it keeps either way, as before.
    let probe_uv  = world_to_sdf_uv(probe_center_world, camera_params.view_proj, camera_params.inv_sdf_scale);
    let probe_sdf = bilinear_sample_r(sdf_in, sdf_in_sampler, probe_uv);
    let interior  = sdf_interior(sdf_in, sdf_in_sampler, probe_uv);
    let fade      = interior_fade(probe_sdf, interior);
    let is_inside = probe_sdf < 0.0;
    // Inside a cave, the skylight is all there is: no push, no march.
    if (fade <= 0.0) {
        store_probe(tile_xy, frame_index, halton_jitter, probe_irradiance);
        return;
    }
    let lit_at    = lit_from(probe_center_world, probe_sdf, sdf_in, sdf_in_sampler, camera_params);

    // Compute direct irradiance from lights in the current frame.
    // Probes inside occluders will have all marches fail naturally, so no
    // explicit SDF gate is needed — and avoiding the hard dist > 0 threshold
    // prevents boundary probes from flickering between skylight-only and
    // skylight+lights as the occluder moves through the probe grid.
    for (var i: i32 = 0; i < i32(lights_source_buffer.count); i++) {

        let light = lights_source_buffer.data[i];

        // Cone (spotlight) shaping. Computed before the raymarch so a narrow
        // cone skips the expensive 32-step march for every probe outside it.
        let to_light = probe_center_world - light.center;
        let to_probe = normalize(to_light);
        // Linear ramp from 0 at the outer rim (cone_cos) to 1 at the inner edge
        // (cone_cos_inner). The epsilon guard keeps a hard rim (the two equal)
        // well-defined.
        //
        // An inner edge at PI (cone_cos_inner = cos(PI) = -1, any omni light)
        // has every direction inside it, so the ramp is skipped. Running it
        // would put a rim straight behind cone_dir, where the cosine is flat:
        // the 1e-4 epsilon there spans 0.8 degrees, a thin dark streak behind
        // the light.
        var cone = 1.0;
        if (light.cone_cos_inner > -1.0) {
            let cone_denom = max(light.cone_cos_inner - light.cone_cos, 1e-4);
            cone = clamp((dot(to_probe, light.cone_dir) - light.cone_cos) / cone_denom, 0.0, 1.0);
        }

        // Near-field fade: mute the light near its apex, where each probe spans a
        // large angle and the cone gradient aliases worst. Radius is in world
        // units (fixed physical size, independent of zoom). cone_near_fade = 0
        // leaves the light unchanged (full circle / omni unaffected).
        cone *= smoothstep(0.0, max(light.cone_near_fade, 1e-6), length(to_light));

        if (cone <= 0.0) {
            continue;
        }

        var march_from = lit_at;
        var light_fade = 1.0;
        if (is_inside) {
            let exit = exit_toward(probe_center_world, light.center, sdf_in, sdf_in_sampler, camera_params);
            march_from = exit.xy;
            light_fade = interior_fade(-exit.z, interior);
        }

        let ray_result = raymarch_primary(
            march_from,
            light.center,
            32,
            sdf_in,
            sdf_in_sampler,
            camera_params,
            0.3
        );

        let att = light_attenuation_r_two(
            probe_center_world,
            light.center,
            light.falloff.x,
            light.falloff.y,
            light.falloff.z,
        );

        if (ray_result.success > 0) {
            probe_irradiance += light.color * att * light.intensity * cone * light_fade;
        }
    }

    store_probe(tile_xy, frame_index, halton_jitter, probe_irradiance);
}