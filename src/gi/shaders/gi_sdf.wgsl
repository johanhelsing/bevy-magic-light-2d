#import bevy_magic_light_2d::gi_types::LightOccluderBuffer
#import bevy_magic_light_2d::gi_math::quat_mul
#import bevy_magic_light_2d::gi_camera::{CameraParams, sdf_uv_to_world}
#import bevy_magic_light_2d::custom_sdf::custom_sdf

@group(0) @binding(0) var<uniform> camera_params:         CameraParams;
@group(0) @binding(1) var<storage> light_occluder_buffer: LightOccluderBuffer;
// x the distance; y, z and w how the nearest occluder lights its interior
// (`lit_depth`, `lit_floor`, `lit_facing`).
@group(0) @binding(2) var          sdf_out:               texture_storage_2d<rgba16float, write>;

// A box of half size `h_extent` with its corners rounded by `radius`: a
// box at 0, a circle or a capsule when the radius is half the narrow side.
fn sdf_aabb_occluder(p: vec2<f32>, occluder_i: i32) -> f32 {
    let occluder = light_occluder_buffer.data[occluder_i];
    let local_p = quat_mul(occluder.rotation, vec3<f32>(occluder.center - p, 0.0)).xy;
    // The app's own shape, exact inside its disc; outside, the disc's
    // distance, which the shape's cannot be less than. Its radius is an
    // inset: the shadow starts that far inside the shape, so the drawn
    // face is lit.
    if (occluder.kind != 0u) {
        let reach = length(local_p);
        if (reach > occluder.h_extent.x) {
            return reach - occluder.h_extent.x + occluder.radius;
        }
        return custom_sdf(-local_p, occluder.kind, occluder.params_a, occluder.params_b) + occluder.radius;
    }
    let d        = abs(local_p) - (occluder.h_extent - vec2<f32>(occluder.radius));
    let d_max    = max(d, vec2<f32>(0.0));
    let d_o      = length(d_max);
    let d_i      = min(max(d.x, d.y), 0.0);
    return d_o + d_i - occluder.radius;
}

fn round_merge(s1: f32, s2: f32, r: f32) -> f32 {
    var intersection_space = vec2<f32>(s1 - r, s1 - r); // s1, s1 is intended
        intersection_space = min(intersection_space, vec2<f32>(0.0));
    let inside_distance    = -length(intersection_space);
    let simple_union       = min(s1, s2);
    let outside_distance   = max(simple_union, r);
    return inside_distance + outside_distance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
     let texel_pos  = vec2<i32>(invocation_id.xy);
     let dims = textureDimensions(sdf_out);
     let uv = (vec2<f32>(texel_pos) + 0.5) / vec2<f32>(dims);

     let world_pose = sdf_uv_to_world(uv,
        camera_params.inverse_view_proj,
        camera_params.sdf_scale);
     // Finite in the half-float texture: 1e10 would store as infinity, and
     // a bilinear sample between two infinities at a texel center is NaN.
     var sdf_merged = 6e4;
     var nearest = -1;
     for (var i: i32 = 0; i < i32(light_occluder_buffer.count); i++) {
        let d = sdf_aabb_occluder(world_pose.xy, i);
        if (d < sdf_merged) {
            sdf_merged = d;
            nearest = i;
        }
     }
     var lit = vec3<f32>(0.0);
     if (nearest >= 0) {
        let occluder = light_occluder_buffer.data[nearest];
        lit = vec3<f32>(occluder.lit_depth, occluder.lit_floor, occluder.lit_facing);
     }

    textureStore(sdf_out, texel_pos, vec4<f32>(sdf_merged, lit.x, lit.y, lit.z));
}