use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

use crate::gi::constants::GI_SCREEN_PROBE_SIZE;
use crate::gi::types::OmniLightSource2D;

#[rustfmt::skip]
#[derive(Default, Clone, ShaderType)]
pub struct GpuOmniLightSource {
    pub center:         Vec2,
    pub intensity:      f32,
    pub color:          Vec3,
    pub falloff:        Vec3,
    /// Normalized cone axis (spotlight direction).
    pub cone_dir:       Vec2,
    /// Cosine of the cone half-angle (outer rim). `cos(PI) = -1` for a full
    /// circle, so the cone test passes everywhere by default.
    pub cone_cos:       f32,
    /// Cosine of the inner edge (`half_angle - softness`); the cone fades from
    /// full at `cone_cos_inner` to zero at `cone_cos`.
    pub cone_cos_inner: f32,
    /// Near-field fade radius in world units (0 = off).
    pub cone_near_fade: f32,
}

impl GpuOmniLightSource
{
    pub fn new(light: OmniLightSource2D, center: Vec2) -> Self
    {
        let color: Srgba = light.color.into();
        let cone_dir = light.cone_direction.try_normalize().unwrap_or(Vec2::X);
        let half_angle = light.cone_half_angle.clamp(0.0, core::f32::consts::PI);
        let inner = (half_angle - light.cone_softness.max(0.0)).max(0.0);
        Self {
            center,
            intensity: light.intensity,
            color: color.to_vec3(),
            falloff: light.falloff,
            cone_dir,
            // Larger angle → smaller cosine, so the outer rim has the smaller
            // value and cone_cos_inner >= cone_cos.
            cone_cos: half_angle.cos(),
            cone_cos_inner: inner.cos(),
            cone_near_fade: light.cone_near_fade.max(0.0),
        }
    }
}

#[rustfmt::skip]
#[derive(Default, Clone, ShaderType)]
pub struct GpuLightSourceBuffer {
    pub count: u32,
    #[shader(size(runtime))]
    pub data:  Vec<GpuOmniLightSource>,
}

#[rustfmt::skip]
#[derive(Default, Clone, ShaderType)]
pub struct GpuLightOccluder2D {
    pub center: Vec2,
    pub rotation: Vec4,
    pub h_extent: Vec2,
}

#[rustfmt::skip]
#[derive(Default, Clone, ShaderType)]
pub struct GpuLightOccluderBuffer {
    pub count: u32,
    #[shader(size(runtime))]
    pub data:  Vec<GpuLightOccluder2D>,
}

#[rustfmt::skip]
#[derive(Default, Clone, ShaderType)]
pub struct GpuCameraParams {
    pub screen_size:       Vec2,
    pub screen_size_inv:   Vec2,
    pub view_proj:         Mat4,
    pub inverse_view_proj: Mat4,
    pub sdf_scale:         Vec2,
    pub inv_sdf_scale:     Vec2,
    /// World units per screen pixel. Allows shaders to scale constants
    /// that were originally tuned for pixel-scale (1 world unit = 1 pixel).
    pub pixel_world_size:  Vec2,
}

#[rustfmt::skip]
#[derive(Clone, ShaderType, Debug)]
pub struct GpuLightPassParams {
    pub frame_counter:          i32,
    pub probe_size:             i32,
    pub probe_atlas_cols:       i32,
    pub probe_atlas_rows:       i32,
    pub skylight_color:         Vec3,

    pub reservoir_size:              u32,
    pub smooth_kernel_size_h:        u32,
    pub smooth_kernel_size_w:        u32,
    pub direct_light_contrib:        f32,
    pub indirect_light_contrib:      f32,
    pub indirect_rays_per_sample:    i32,
    pub indirect_rays_radius_factor: f32,
}

impl Default for GpuLightPassParams
{
    fn default() -> Self
    {
        Self {
            frame_counter:    0,
            probe_size:       0,
            probe_atlas_cols: 0,
            probe_atlas_rows: 0,
            skylight_color:   Vec3::new(0.003, 0.0078, 0.058) / 100.0,

            reservoir_size:         16,
            smooth_kernel_size_h:   3,
            smooth_kernel_size_w:   3,
            direct_light_contrib:   0.2,
            indirect_light_contrib: 0.8,

            indirect_rays_per_sample:    64,
            indirect_rays_radius_factor: 3.0,
        }
    }
}

#[rustfmt::skip]
#[derive(Clone, ShaderType, Default)]
pub struct GpuProbeData {
    pub camera_pose: Vec2,
}

#[rustfmt::skip]
#[derive(Clone, ShaderType)]
pub struct GpuProbeDataBuffer {
    pub count: u32,
    #[shader(size(runtime))]
    pub data:  Vec<GpuProbeData>,
}

impl Default for GpuProbeDataBuffer
{
    fn default() -> Self
    {
        const MAX_PROBES: u32 = (GI_SCREEN_PROBE_SIZE * GI_SCREEN_PROBE_SIZE) as u32;
        Self {
            count: MAX_PROBES,
            data:  vec![
                GpuProbeData {
                    camera_pose: Vec2::ZERO,
                };
                MAX_PROBES as usize
            ],
        }
    }
}

#[rustfmt::skip]
#[derive(Clone, ShaderType, Default)]
pub struct GpuSkylightMaskData {
    pub center:   Vec2,
    pub h_extent: Vec2,
}

impl GpuSkylightMaskData
{
    pub fn new(center: Vec2, h_extent: Vec2) -> Self
    {
        Self { center, h_extent }
    }
}

#[rustfmt::skip]
#[derive(Clone, ShaderType, Default)]
pub struct GpuSkylightMaskBuffer {
    pub count: u32,
    #[shader(size(runtime))]
    pub data: Vec<GpuSkylightMaskData>,
}
