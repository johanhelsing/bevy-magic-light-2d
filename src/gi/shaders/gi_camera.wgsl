#define_import_path bevy_magic_light_2d::gi_camera

struct CameraParams {
    screen_size:         vec2<f32>,
    screen_size_inv:     vec2<f32>,
    view_proj:           mat4x4<f32>,
    inverse_view_proj:   mat4x4<f32>,
    sdf_scale:           vec2<f32>,
    inv_sdf_scale:       vec2<f32>,
    pixel_world_size:    vec2<f32>,
}

fn screen_to_ndc(
    screen_pose:     vec2<i32>,
    screen_size:     vec2<f32>,
    screen_size_inv: vec2<f32>) -> vec2<f32> {
    let screen_pose_f32 = vec2<f32>(0.0, screen_size.y)
                        + vec2<f32>(f32(screen_pose.x), f32(-screen_pose.y));
    return (screen_pose_f32 * screen_size_inv) * 2.0 - 1.0;
}

fn ndc_to_screen(ndc: vec2<f32>, screen_size: vec2<f32>) -> vec2<i32> {
    let screen_pose_f32 = (ndc + 1.0) * 0.5 * screen_size;
    return vec2<i32>(
        i32(screen_pose_f32.x),
        i32(screen_size.y - screen_pose_f32.y),
    );
}

fn screen_to_world(
    screen_pose:       vec2<i32>,
    screen_size:       vec2<f32>,
    inverse_view_proj: mat4x4<f32>,
    screen_size_inv:   vec2<f32>) -> vec2<f32> {
    return (inverse_view_proj * vec4<f32>(screen_to_ndc(screen_pose, screen_size, screen_size_inv), 0.0, 1.0)).xy;
}

// World-space offset of a probe's Halton-jittered sample point from the
// top-left corner of its probe tile. `jitter` is in [0,1)^2 and walks the
// sample across the tile over the reservoir, so the samples average to the
// tile centre.
//
// The y component is negated because the two spaces disagree on which way is
// up: the tile origin comes from `screen_to_world` of a screen pose whose y
// grows downwards, while the offset is added in world space, where y grows
// upwards. Adding it unnegated walks the sample a full tile *above* the tile
// it is stored for, and every light then renders one probe tile too low.
fn probe_jitter_offset_world(
    jitter:           vec2<f32>,
    probe_size:       f32,
    pixel_world_size: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(jitter.x, -jitter.y) * probe_size * pixel_world_size;
}

fn world_to_ndc(
    world_pose:  vec2<f32>,
    view_proj:   mat4x4<f32>) -> vec2<f32> {
    return (view_proj * vec4<f32>(world_pose, 0.0, 1.0)).xy;
}

fn world_to_screen(
    world_pose:  vec2<f32>,
    screen_size: vec2<f32>,
    view_proj:   mat4x4<f32>) -> vec2<i32> {
    return ndc_to_screen(world_to_ndc(world_pose, view_proj), screen_size);
}

fn world_to_sdf_uv(world_pose: vec2<f32>, view_proj: mat4x4<f32>, inv_sdf_scale: vec2<f32>) -> vec2<f32> {
    let ndc = world_to_ndc(world_pose, view_proj);
    let ndc_sdf = ndc * inv_sdf_scale;
    let uv = (ndc_sdf + 1.0) * 0.5;
    let y = 1.0 - uv.y;
    return vec2<f32>(uv.x, y);
}

fn sdf_uv_to_world(uv_in: vec2<f32>, inverse_view_proj: mat4x4<f32>, sdf_scale: vec2<f32>) -> vec2<f32> {
    let y = 1.0 - uv_in.y;
    let uv = vec2<f32>(uv_in.x, y);
    let ndc_sdf = (uv * 2.0) - 1.0;
    let ndc = ndc_sdf * sdf_scale;
    return (inverse_view_proj * vec4<f32>(ndc, 0.0, 1.0)).xy;
}

// The 2x2 texels a bilinear sample at `uv` reads and the weights within
// them: `uv` the center of that footprint, to gather at, `f` the fraction
// of the way from its first texel to its second. Gathering at the sample's
// own uv leaves the footprint to the sampler, which rounds the coordinate
// to fixed point before it floors it, while the weights are floored in
// f32: a sample a rounding error short of a texel center then weights the
// texel after the one the sampler took, and reads the value one texel
// over. The irradiance filter samples at pixel corners, and at the shipped
// scale every fourth of those sits on a texel center, so the seam ran
// along a rock's rim as teeth, each a texel long. Gathered at the
// footprint's center, the sampler is half a texel from any rounding
// boundary and agrees with the weights.
struct Footprint {
    uv: vec2<f32>,
    f:  vec2<f32>,
}

fn footprint(uv: vec2<f32>, dims: vec2<u32>) -> Footprint {
    let dims_f = vec2<f32>(dims);
    let p  = uv * dims_f - 0.5;
    let p0 = floor(p);
    return Footprint((p0 + 1.0) / dims_f, p - p0);
}

fn bilinear_filter(texels: vec4<f32>, f: vec2<f32>) -> f32 {
    // texels.x = -u, +v
    // texels.y = +u, +v,
    // texels.z = +u, -v,
    // texels.w = -u, -v
    return mix(mix(texels.w, texels.z, f.x), mix(texels.x, texels.y, f.x), f.y);
}

fn bilinear_sample_r(t: texture_2d<f32>, s: sampler, uv: vec2<f32>) -> f32 {
    let fp = footprint(uv, textureDimensions(t));
    return bilinear_filter(textureGather(0, t, s, fp.uv), fp.f);
}

// How the nearest occluder lights its interior at `uv` of the distance
// texture: x how far light reaches in, meters, y the floor (`LightOccluder2D`).
fn sdf_interior(t: texture_2d<f32>, s: sampler, uv: vec2<f32>) -> vec2<f32> {
    let fp = footprint(uv, textureDimensions(t));
    let depth = bilinear_filter(textureGather(1, t, s, fp.uv), fp.f);
    let floor = bilinear_filter(textureGather(2, t, s, fp.uv), fp.f);
    return vec2<f32>(depth, floor);
}

// Whether the nearest occluder lights its interior through itself toward
// each lamp at `uv` (`LightOccluder2D::lit_through`), above 0.5.
fn sdf_through(t: texture_2d<f32>, s: sampler, uv: vec2<f32>) -> f32 {
    let fp = footprint(uv, textureDimensions(t));
    return bilinear_filter(textureGather(3, t, s, fp.uv), fp.f);
}

fn bilinear_sample_rgba(t: texture_2d<f32>, s: sampler, uv: vec2<f32>) -> vec4<f32> {
    let fp = footprint(uv, textureDimensions(t));
    let r = bilinear_filter(textureGather(0, t, s, fp.uv), fp.f);
    let g = bilinear_filter(textureGather(1, t, s, fp.uv), fp.f);
    let b = bilinear_filter(textureGather(2, t, s, fp.uv), fp.f);
    let a = bilinear_filter(textureGather(3, t, s, fp.uv), fp.f);
    return vec4<f32>(r, g, b, a);
}
