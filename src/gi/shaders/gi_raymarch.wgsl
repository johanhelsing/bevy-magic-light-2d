#define_import_path bevy_magic_light_2d::gi_raymarch

#import bevy_magic_light_2d::gi_math::{fast_normalize_2d, distance_squared, hash}
#import bevy_magic_light_2d::gi_camera::{CameraParams, sdf_uv_to_world, world_to_sdf_uv, bilinear_sample_r}

// How close to an occluder a march may pass before it counts as blocked.
// A march that starts nearer than this fails on its first step, so a caller
// starting on a boundary moves its origin out past it first.
fn min_march_sdf(camera_params: CameraParams) -> f32 {
    return camera_params.pixel_world_size.x * 0.6;
}

// Where a point inside an occluder, or nearer its boundary than the march's
// minimum distance, gathers its light: itself pushed out along the SDF
// gradient to that distance. A point already clear of every boundary is
// its own. Inside a shape the distance is only an estimate (a union's is
// exact outside, not in), so the push is taken again from where it lands,
// a few times, until the point is clear.
fn lit_from(
    p:             vec2<f32>,
    sdf_at_p:      f32,
    sdf:           texture_2d<f32>,
    sdf_sampler:   sampler,
    camera_params: CameraParams,
) -> vec2<f32> {
    let min_sdf = min_march_sdf(camera_params);
    let h = camera_params.pixel_world_size.x;
    var at = p;
    var d = sdf_at_p;
    for (var i = 0; i < 4; i++) {
        if d >= min_sdf {
            return at;
        }
        let dx = bilinear_sample_r(sdf, sdf_sampler,
            world_to_sdf_uv(at + vec2<f32>(h, 0.0), camera_params.view_proj, camera_params.inv_sdf_scale)) - d;
        let dy = bilinear_sample_r(sdf, sdf_sampler,
            world_to_sdf_uv(at + vec2<f32>(0.0, h), camera_params.view_proj, camera_params.inv_sdf_scale)) - d;
        let grad = vec2<f32>(dx, dy);
        let len = length(grad);
        if len <= 0.0 {
            return at;
        }
        at += grad / len * (min_sdf - d + h * 0.5);
        d = bilinear_sample_r(sdf, sdf_sampler,
            world_to_sdf_uv(at, camera_params.view_proj, camera_params.inv_sdf_scale));
    }
    return at;
}

// Where a point inside an occluder leaves it on its way to `toward`, and how
// far it travelled inside: xy the first point clear of the boundary by the
// march's minimum distance, z the distance inside. Inside, the distance to the
// nearest surface is never more than the distance to where the ray leaves, so
// stepping by it never steps out past the exit (sphere tracing, from inside).
// A point `toward` inside the same occluder ends the walk there.
fn exit_toward(
    p:             vec2<f32>,
    toward:        vec2<f32>,
    sdf:           texture_2d<f32>,
    sdf_sampler:   sampler,
    camera_params: CameraParams,
) -> vec3<f32> {
    let min_sdf = min_march_sdf(camera_params);
    let px      = camera_params.pixel_world_size.x;
    let reach   = distance(p, toward);
    let dir     = (toward - p) / max(reach, 1e-6);
    var walked  = 0.0;
    var inside  = 0.0;
    for (var i = 0; i < 24; i++) {
        let at = p + dir * walked;
        let d = bilinear_sample_r(sdf, sdf_sampler,
            world_to_sdf_uv(at, camera_params.view_proj, camera_params.inv_sdf_scale));
        if !(d < min_sdf) || walked >= reach {
            break;
        }
        let step = max(abs(d), px);
        if d < 0.0 {
            inside += step;
        }
        walked += step;
    }
    return vec3<f32>(p + dir * min(walked, reach), inside);
}

// How much of its surface's light a point `sdf_at_p` into an occluder gets:
// faded by depth over the occluder's `lit_depth`, never below its
// `lit_floor`; 1 outside. An occluder with no depth gives its floor alone.
fn interior_fade(sdf_at_p: f32, interior: vec2<f32>) -> f32 {
    // Written so a NaN distance counts as outside, never as a cave.
    if !(sdf_at_p < 0.0) {
        return 1.0;
    }
    let floor = clamp(interior.y, 0.0, 1.0);
    if interior.x <= 0.0 {
        return floor;
    }
    return max(floor, exp(sdf_at_p / interior.x));
}

struct RayMarchResult {
    success:  i32,      //
    step: i32,          // steps
    pose: vec2<f32>,    // curr spot
}

fn raymarch(
    in_ray_origin:      vec2<f32>,
    in_ray_target:      vec2<f32>,
    max_steps:          i32,
    sdf:                texture_2d<f32>,
    sdf_sampler:        sampler,
    camera_params:      CameraParams,
    rm_jitter_contrib:  f32,
) -> RayMarchResult {

    var ray_target  = in_ray_target;
    var ray_origin  = in_ray_origin;
    let target_uv   = world_to_sdf_uv(ray_target, camera_params.view_proj, camera_params.inv_sdf_scale);
    let target_dist = bilinear_sample_r(sdf, sdf_sampler, target_uv);

    if (target_dist < 0.0) {
        let t = ray_target;
        ray_target = ray_origin;
        ray_origin = t;
    }

    let ray_direction          = fast_normalize_2d(ray_target - ray_origin);
    let stop_at                = distance_squared(ray_origin, ray_target);

    var ray_progress:   f32    = 0.0;
    var h                      = vec2<f32>(0.0);
    var h_prev                 = h;
    let min_sdf                = min_march_sdf(camera_params);
    var inside                 = true;
    let max_inside_dist        = 20.0;
    let max_inside_dist_sq     = max_inside_dist * max_inside_dist;

    for (var i: i32 = 0; i < max_steps; i++) {

        h_prev = h;
        h = ray_origin + ray_progress * ray_direction;

        if ((ray_progress * ray_progress >= stop_at) || (inside && (ray_progress * ray_progress > max_inside_dist))) {
            return RayMarchResult(1, i, h_prev);
        }


        let uv = world_to_sdf_uv(h, camera_params.view_proj, camera_params.inv_sdf_scale);
        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
            return RayMarchResult(0, i, h_prev);
        }

        let scene_dist = bilinear_sample_r(sdf, sdf_sampler, uv);
        if ((scene_dist <= min_sdf && !inside)) {
            return RayMarchResult(0, i, h);
        }
        if (scene_dist > 0.0) {
            inside = false;
        }
        let ray_travel = max(abs(scene_dist), camera_params.pixel_world_size.x);

        if (rm_jitter_contrib > 0.0) {
            // Jitter step.
            let jitter = hash(h);
            ray_progress += ray_travel * (1.0 - rm_jitter_contrib) + rm_jitter_contrib * ray_travel * jitter;
        } else {
            ray_progress += ray_travel;
        }
    }

    return RayMarchResult(0, max_steps, h);
}

fn raymarch_primary(
    in_ray_origin:      vec2<f32>,
    in_ray_target:      vec2<f32>,
    max_steps:          i32,
    sdf:                texture_2d<f32>,
    sdf_sampler:        sampler,
    camera_params:      CameraParams,
    rm_jitter_contrib:  f32,
) -> RayMarchResult {

    var ray_target  = in_ray_target;
    var ray_origin  = in_ray_origin;

    let ray_direction          = normalize(ray_target - ray_origin);
    let stop_at                = distance_squared(ray_origin, ray_target);

    var ray_progress:   f32    = 0.0;
    var h                      = vec2<f32>(0.0);
    var h_prev                 = h;
    let min_sdf                = min_march_sdf(camera_params);

    for (var i: i32 = 0; i < max_steps; i++) {

        h_prev = h;
        h = ray_origin + ray_progress * ray_direction;

        if ray_progress * ray_progress >= stop_at {
            return RayMarchResult(1, i, h_prev);
        }


        let uv = world_to_sdf_uv(h, camera_params.view_proj, camera_params.inv_sdf_scale);
        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
            return RayMarchResult(0, i, h_prev);
        }

        let scene_dist = bilinear_sample_r(sdf, sdf_sampler, uv);
        if scene_dist <= min_sdf {
            return RayMarchResult(0, i, h);
        }

        let ray_travel = max(abs(scene_dist), 0.0);

        ray_progress += ray_travel * (1.0 - rm_jitter_contrib) + rm_jitter_contrib * ray_travel * hash(h);
   }

    return RayMarchResult(0, max_steps, h);
}


fn raymarch_bounce(
    in_ray_origin:      vec2<f32>,
    in_ray_target:      vec2<f32>,
    max_steps:          i32,
    sdf:                texture_2d<f32>,
    sdf_sampler:        sampler,
    camera_params:      CameraParams,
    rm_jitter_contrib:  f32,
) -> RayMarchResult {

    var ray_target  = in_ray_target;
    var ray_origin  = in_ray_origin;

    let target_uv   = world_to_sdf_uv(ray_target, camera_params.view_proj, camera_params.inv_sdf_scale);
    let target_dist = bilinear_sample_r(sdf, sdf_sampler, target_uv);

    if (target_dist < 0.0) {
        let temp = ray_target;
        ray_target = ray_origin;
        ray_origin = temp;
    }

    let ray_direction          = fast_normalize_2d(ray_target - ray_origin);
    let stop_at                = distance_squared(ray_origin, ray_target);

    var ray_progress:   f32    = 0.0;
    var h                      = vec2<f32>(0.0);
    var h_prev                 = h;
    let min_sdf                = min_march_sdf(camera_params);

    for (var i: i32 = 0; i < max_steps; i++) {

        h_prev = h;
        h = ray_origin + ray_progress * ray_direction;

        if ray_progress * ray_progress >= stop_at {
            return RayMarchResult(1, i, h_prev);
        }

        let uv = world_to_sdf_uv(h, camera_params.view_proj, camera_params.inv_sdf_scale);
        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
            return RayMarchResult(0, i, h_prev);
        }

        let scene_dist = bilinear_sample_r(sdf, sdf_sampler, uv);
        if  scene_dist <= min_sdf {
            return RayMarchResult(0, i, h);
        }

        let ray_travel = max(abs(scene_dist), camera_params.pixel_world_size.x);

        ray_progress += ray_travel * (1.0 - rm_jitter_contrib)
                      + rm_jitter_contrib * ray_travel * hash(h);
    }

    return RayMarchResult(0, max_steps, h);
}