#import bevy_magic_light_2d::gi_types::LightOccluderBuffer
#import bevy_magic_light_2d::gi_math::quat_mul
#import bevy_magic_light_2d::gi_camera::{CameraParams, sdf_uv_to_world}
#import bevy_magic_light_2d::custom_sdf::custom_sdf

@group(0) @binding(0) var<uniform> camera_params:         CameraParams;
@group(0) @binding(1) var<storage> light_occluder_buffer: LightOccluderBuffer;
// x the distance; y, z and w how the nearest occluder lights its interior
// (`lit_depth`, `lit_floor`, `lit_through`).
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

// How far a texel can be from an occluder's center and still be inside it,
// and the most its distance can exceed the distance to that center. A
// distance to a shape is 1-Lipschitz, so between the two the occluder's
// distance at `p` lies in [|p - center| - inner, |p - center| + outer].
struct Reach {
    inner: f32,
    outer: f32,
}

fn occluder_reach(occluder_i: u32) -> Reach {
    let occluder = light_occluder_buffer.data[occluder_i];
    if (occluder.kind != 0u) {
        // The shape lies inside its disc, and the pass adds the inset.
        return Reach(occluder.h_extent.x - occluder.radius, occluder.h_extent.x + occluder.radius);
    }
    // The box's distance at its own center is the most it adds to the
    // distance to the center anywhere.
    let d = -(occluder.h_extent - vec2<f32>(occluder.radius));
    let at_center = length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0) - occluder.radius;
    return Reach(length(max(occluder.h_extent, vec2<f32>(occluder.radius))), at_center);
}

fn texel_world(texel: vec2<i32>, dims: vec2<u32>) -> vec2<f32> {
    let uv = (vec2<f32>(texel) + 0.5) / vec2<f32>(dims);
    return sdf_uv_to_world(uv, camera_params.inverse_view_proj, camera_params.sdf_scale).xy;
}

// The workgroup's tile: the most its nearest occluder can be away, as the
// bits of a non-negative float, so `atomicMin` orders them.
var<workgroup> tile_bound: atomic<u32>;
// The occluders of one batch that can be nearest somewhere in the tile.
var<workgroup> kept:       array<u32, 64>;
var<workgroup> kept_count: atomic<u32>;

// Every texel measured against every occluder costs texels times
// occluders, and a field of rocks is over a thousand of them. So each
// 8x8 tile first bounds how far its nearest occluder can be (the least,
// over occluders, of the most each can be away anywhere in the tile), then
// measures only the occluders that can come closer than that somewhere in
// it. The others cannot be nearest at any of its texels, so the distance
// and the nearest occluder are what measuring all of them gives (ties go
// to the lower index, as before).
@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id)   invocation_id: vec3<u32>,
    @builtin(workgroup_id)           workgroup_id:  vec3<u32>,
    @builtin(local_invocation_index) lane:          u32,
) {
    let texel_pos  = vec2<i32>(invocation_id.xy);
    let dims       = textureDimensions(sdf_out);
    let world_pose = texel_world(texel_pos, dims);

    // The tile as a disc around the centers of its corner texels.
    let first    = vec2<i32>(workgroup_id.xy * 8u);
    let corner_a = texel_world(first, dims);
    let corner_b = texel_world(first + vec2<i32>(7, 0), dims);
    let corner_c = texel_world(first + vec2<i32>(0, 7), dims);
    let corner_d = texel_world(first + vec2<i32>(7, 7), dims);
    let tile_center = (corner_a + corner_b + corner_c + corner_d) * 0.25;
    let tile_radius = max(
        max(distance(tile_center, corner_a), distance(tile_center, corner_b)),
        max(distance(tile_center, corner_c), distance(tile_center, corner_d)),
    );

    // Finite in the half-float texture: 1e10 would store as infinity, and
    // a bilinear sample between two infinities at a texel center is NaN.
    let far = 6e4;
    if (lane == 0u) {
        atomicStore(&tile_bound, bitcast<u32>(far));
    }
    workgroupBarrier();
    let count = light_occluder_buffer.count;
    for (var i = lane; i < count; i += 64u) {
        let to_center = distance(tile_center, light_occluder_buffer.data[i].center);
        let most = to_center + tile_radius + occluder_reach(i).outer;
        atomicMin(&tile_bound, bitcast<u32>(max(most, 0.0)));
    }
    workgroupBarrier();
    let bound = bitcast<f32>(atomicLoad(&tile_bound));

    var sdf_merged = far;
    var nearest = -1;
    for (var batch = 0u; batch < count; batch += 64u) {
        if (lane == 0u) {
            atomicStore(&kept_count, 0u);
        }
        workgroupBarrier();
        let i = batch + lane;
        if (i < count) {
            let to_center = distance(tile_center, light_occluder_buffer.data[i].center);
            if (to_center - tile_radius - occluder_reach(i).inner <= bound) {
                kept[atomicAdd(&kept_count, 1u)] = i;
            }
        }
        workgroupBarrier();
        let n = atomicLoad(&kept_count);
        for (var k = 0u; k < n; k++) {
            let j = i32(kept[k]);
            let d = sdf_aabb_occluder(world_pose, j);
            if (d < sdf_merged || (d == sdf_merged && j < nearest)) {
                sdf_merged = d;
                nearest = j;
            }
        }
        workgroupBarrier();
    }
    var lit = vec3<f32>(0.0);
    if (nearest >= 0) {
        let occluder = light_occluder_buffer.data[nearest];
        lit = vec3<f32>(occluder.lit_depth, occluder.lit_floor, occluder.lit_through);
    }

    textureStore(sdf_out, texel_pos, vec4<f32>(sdf_merged, lit.x, lit.y, lit.z));
}
