use std::path::Path;

use bevy::asset::io::AssetSourceId;
use bevy::asset::AssetPath;
use bevy::prelude::*;
use bevy::render::render_resource::{StorageBuffer, UniformBuffer};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::Extract;
use rand::Rng;

use crate::gi::constants::{
    COARSE_SDF_LIGHT_FADE,
    COARSE_SDF_VIEW_SCALE,
    GI_SCREEN_PROBE_SIZE,
    SDF_CULL_MARGIN,
    SDF_VIEW_SCALE,
};
use crate::gi::custom_sdf::CustomSdfs;
use crate::gi::occluder_merge::{OccluderBox, RowMerger};
use crate::gi::resource::ComputedTargetSizes;
use crate::gi::types::{LightOccluder2D, OmniLightSource2D, SkylightLight2D, SkylightMask2D};
use crate::gi::types_gpu::{
    GpuCameraParams,
    GpuLightOccluder2D,
    GpuLightOccluderBuffer,
    GpuLightPassParams,
    GpuLightSourceBuffer,
    GpuOmniLightSource,
    GpuProbeDataBuffer,
    GpuSkylightMaskBuffer,
    GpuSkylightMaskData,
};
use crate::prelude::BevyMagicLight2DSettings;
use crate::FloorCamera;

#[rustfmt::skip]
#[derive(Default, Resource)]
pub(crate) struct EmbeddedShaderDependencies {
    loaded_shaders: Vec<Handle<Shader>>,
}

#[rustfmt::skip]
pub(crate) fn system_load_embedded_shader_dependencies(
    mut embedded_shader_deps: ResMut<EmbeddedShaderDependencies>,
    asset_server: Res<AssetServer>,
) {
    embedded_shader_deps.loaded_shaders.push(load_embedded_shader(&asset_server, "gi_attenuation.wgsl"));
    embedded_shader_deps.loaded_shaders.push(load_embedded_shader(&asset_server, "gi_camera.wgsl"));
    embedded_shader_deps.loaded_shaders.push(load_embedded_shader(&asset_server, "gi_halton.wgsl"));
    embedded_shader_deps.loaded_shaders.push(load_embedded_shader(&asset_server, "gi_math.wgsl"));
    embedded_shader_deps.loaded_shaders.push(load_embedded_shader(&asset_server, "gi_post_processing.wgsl"));
    embedded_shader_deps.loaded_shaders.push(load_embedded_shader(&asset_server, "gi_raymarch.wgsl"));
    embedded_shader_deps.loaded_shaders.push(load_embedded_shader(&asset_server, "gi_types.wgsl"));
}

pub(crate) fn load_embedded_shader(asset_server: &AssetServer, shader_file: &str)
    -> Handle<Shader>
{
    let source = AssetSourceId::from("embedded");
    let path = Path::new("bevy_magic_light_2d").join("gi/shaders/");
    asset_server.load(AssetPath::from_path(&path.join(shader_file)).with_source(&source))
}

#[rustfmt::skip]
#[derive(Default, Resource)]
pub struct LightPassPipelineAssets {
    pub camera_params:          UniformBuffer<GpuCameraParams>,
    /// The camera as the coarse SDF sees it: `sdf_scale` is the coarse one.
    pub coarse_camera_params:   UniformBuffer<GpuCameraParams>,
    pub light_pass_params:      UniformBuffer<GpuLightPassParams>,
    pub light_sources:          StorageBuffer<GpuLightSourceBuffer>,
    pub light_occluders:        StorageBuffer<GpuLightOccluderBuffer>,
    pub coarse_light_occluders: StorageBuffer<GpuLightOccluderBuffer>,
    pub probes:                 StorageBuffer<GpuProbeDataBuffer>,
    pub skylight_masks:         StorageBuffer<GpuSkylightMaskBuffer>,
}

impl LightPassPipelineAssets
{
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue)
    {
        self.light_sources.write_buffer(device, queue);
        self.light_occluders.write_buffer(device, queue);
        self.coarse_light_occluders.write_buffer(device, queue);
        self.camera_params.write_buffer(device, queue);
        self.coarse_camera_params.write_buffer(device, queue);
        self.light_pass_params.write_buffer(device, queue);
        self.probes.write_buffer(device, queue);
        self.skylight_masks.write_buffer(device, queue);
    }
}

#[rustfmt::skip]
pub fn system_prepare_pipeline_assets(
    render_device:         Res<RenderDevice>,
    render_queue:          Res<RenderQueue>,
    mut gi_compute_assets: ResMut<LightPassPipelineAssets>,
) {
    gi_compute_assets.write_buffer(&render_device, &render_queue);
}

#[rustfmt::skip]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn system_extract_pipeline_assets(
    res_light_settings:         Extract<Res<BevyMagicLight2DSettings>>,
    res_target_sizes:           Extract<Res<ComputedTargetSizes>>,
    res_custom_sdfs:            Extract<Res<CustomSdfs>>,

    query_lights:               Extract<Query<(&GlobalTransform, &OmniLightSource2D, &InheritedVisibility)>>,
    query_occluders:            Extract<Query<(&LightOccluder2D, &GlobalTransform, &InheritedVisibility)>>,
    query_camera:               Extract<Query<(&Camera, &GlobalTransform), With<FloorCamera>>>,
    query_masks:                Extract<Query<(&GlobalTransform, &SkylightMask2D)>>,
    query_skylight_light:       Extract<Query<&SkylightLight2D>>,

    mut gpu_target_sizes:       ResMut<ComputedTargetSizes>,
    mut gpu_pipeline_assets:    ResMut<LightPassPipelineAssets>,
    mut gpu_frame_counter:      Local<i32>,
    mut occluder_rows:          Local<RowMerger>,
) {
    let light_pass_config = &res_light_settings.light_pass_params;

    *gpu_target_sizes = **res_target_sizes;

    let camera = query_camera.single().ok().map(|(camera, transform)| {
        let view: Mat4 = transform.affine().into();
        (view * camera.clip_from_view().inverse(), camera.clip_from_view() * view.inverse())
    });

    {
        let light_sources = gpu_pipeline_assets.light_sources.get_mut();
        let mut rng = rand::rng();
        light_sources.count = 0;
        light_sources.data.clear();
        for (transform, light_source, visibility) in query_lights.iter() {
            // Lights and occluders are culled by where they are, never by
            // the view's frustum: one off screen still lights or shades it.
            // A light past the coarse SDF lights nothing, so no march toward
            // a light ever leaves it; one near its edge fades on the way out.
            let fade = camera.map_or(1.0, |(_, ndc_from_world)| {
                let ndc = (ndc_from_world * transform.translation().extend(1.0)).xy();
                let edge = (ndc / COARSE_SDF_VIEW_SCALE).abs().max_element();
                1.0 - smoothstep(COARSE_SDF_LIGHT_FADE, 1.0, edge)
            });
            if visibility.get() && fade > 0.0 {
                light_sources.count += 1;
                light_sources.data.push(GpuOmniLightSource::new(
                    OmniLightSource2D {
                        intensity: fade
                            * (light_source.intensity
                                + rng.random_range(-1.0..1.0) * light_source.jitter_intensity),
                        ..*light_source
                    },
                    Vec2::new(
                        transform.translation().x
                            + rng.random_range(-1.0..1.0) * light_source.jitter_translation,
                        transform.translation().y
                            + rng.random_range(-1.0..1.0) * light_source.jitter_translation,
                    ),
                ));
            }
        }
    }

    // The SDF pass measures every texel against every occluder it is given,
    // so each SDF gets only those near the area it covers. A farther one can
    // only shorten march steps near the SDF's edge, never add or remove a hit
    // (a march stops where it leaves the coarse SDF, and goes on in it where
    // it leaves the fine one); the margin keeps even that off. Each row of
    // touching tiles goes as one box (`occluder_merge`).
    let sdf_area = |scale: f32| {
        camera.map(|(world_from_ndc, _)| {
            let corners = [Vec2::new(-1.0, -1.0), Vec2::new(1.0, -1.0), Vec2::new(-1.0, 1.0), Vec2::new(1.0, 1.0)]
                .map(|c| (world_from_ndc * (c * scale).extend(0.0).extend(1.0)).xy());
            let area = Rect::from_corners(corners[0], corners[3]).union(Rect::from_corners(corners[1], corners[2]));
            area.inflate(area.half_size().max_element() * SDF_CULL_MARGIN)
        })
    };
    let near_area = |area: Option<Rect>, center: Vec2, reach: f32| {
        area.is_none_or(|area| (center - center.clamp(area.min, area.max)).length() <= reach)
    };
    let fine_area = sdf_area(SDF_VIEW_SCALE);
    let coarse_area = sdf_area(COARSE_SDF_VIEW_SCALE);

    {
        let near: Vec<OccluderBox> = query_occluders
            .iter()
            .filter(|(occluder, global_transform, visibility)| {
                visibility.get()
                    && near_area(coarse_area, global_transform.translation().xy(), occluder.h_size.length())
            })
            .filter_map(|(occluder, global_transform, _)| {
                // A shape whose shader is not in the module yet casts
                // nothing until it is.
                let kind = match &occluder.sdf {
                    None => 0,
                    Some(shader) => res_custom_sdfs.kind(shader)?,
                };
                Some(OccluderBox {
                    center: global_transform.translation().xy(),
                    rotation: global_transform.rotation(),
                    h_extent: occluder.h_size,
                    radius: occluder.radius,
                    kind,
                    params: occluder.params,
                    lit_depth: occluder.lit_depth,
                    lit_floor: occluder.lit_floor,
                    lit_through: if occluder.lit_through { 1.0 } else { 0.0 },
                })
            })
            .collect();
        let boxes = if res_light_settings.merge_occluder_rows {
            occluder_rows.merge(near).to_vec()
        } else {
            near
        };
        let coarse_occluders = gpu_pipeline_assets.coarse_light_occluders.get_mut();
        coarse_occluders.data.clear();
        coarse_occluders.data.extend(boxes.iter().map(|b| GpuLightOccluder2D {
            center:   b.center,
            rotation: b.rotation.inverse().into(),
            h_extent: b.h_extent,
            radius:   b.radius,
            kind:     b.kind,
            params_a: Vec4::from_slice(&b.params[..4]),
            params_b: Vec4::from_slice(&b.params[4..]),
            lit_depth: b.lit_depth,
            lit_floor: b.lit_floor,
            lit_through: b.lit_through,
        }));
        coarse_occluders.count = coarse_occluders.data.len() as u32;
        let fine: Vec<GpuLightOccluder2D> = boxes
            .iter()
            .zip(&coarse_occluders.data)
            .filter(|(b, _)| near_area(fine_area, b.center, b.h_extent.length()))
            .map(|(_, gpu)| gpu.clone())
            .collect();
        let light_occluders = gpu_pipeline_assets.light_occluders.get_mut();
        light_occluders.count = fine.len() as u32;
        light_occluders.data = fine;
    }

    {
        let skylight_masks = gpu_pipeline_assets.skylight_masks.get_mut();
        skylight_masks.count = 0;
        skylight_masks.data.clear();
        for (transform, mask) in query_masks.iter() {
            skylight_masks.count += 1;
            skylight_masks.data.push(GpuSkylightMaskData::new(
                transform.translation().truncate(),
                mask.h_size,
            ));
        }
    }

    {
        if let Ok((camera, camera_global_transform)) = query_camera.single() {
            let camera_params = gpu_pipeline_assets.camera_params.get_mut();
            let projection = camera.clip_from_view();
            let inverse_projection = projection.inverse();
            let view: Mat4 = camera_global_transform.affine().into();
            let inverse_view = view.inverse();

            camera_params.view_proj = projection * inverse_view;
            camera_params.inverse_view_proj = view * inverse_projection;
            camera_params.screen_size = Vec2::new(
                gpu_target_sizes.primary_target_size.x,
                gpu_target_sizes.primary_target_size.y,
            );
            camera_params.screen_size_inv = Vec2::new(
                1.0 / gpu_target_sizes.primary_target_size.x,
                1.0 / gpu_target_sizes.primary_target_size.y,
            );

            camera_params.sdf_scale     = Vec2::splat(SDF_VIEW_SCALE);
            camera_params.inv_sdf_scale = Vec2::splat(1. / SDF_VIEW_SCALE);

            // Compute world units per pixel from the camera's projection.
            // inverse_view_proj maps NDC to world. NDC spans [-1,1] = 2 units.
            let ndc_top_right = view * inverse_projection * Vec4::new(1.0, 1.0, 0.0, 1.0);
            let ndc_bot_left  = view * inverse_projection * Vec4::new(-1.0, -1.0, 0.0, 1.0);
            let world_extent = (ndc_top_right.xy() - ndc_bot_left.xy()).abs();
            camera_params.pixel_world_size = world_extent / camera_params.screen_size;

            let coarse = GpuCameraParams {
                sdf_scale:     Vec2::splat(COARSE_SDF_VIEW_SCALE),
                inv_sdf_scale: Vec2::splat(1. / COARSE_SDF_VIEW_SCALE),
                ..camera_params.clone()
            };
            *gpu_pipeline_assets.coarse_camera_params.get_mut() = coarse;

            let probes = gpu_pipeline_assets.probes.get_mut();
            probes.data[*gpu_frame_counter as usize].camera_pose =
                camera_global_transform.translation().truncate();
        } else {
            log::warn!("Failed to get camera");
            let probes = gpu_pipeline_assets.probes.get_mut();
            probes.data[*gpu_frame_counter as usize].camera_pose = Vec2::ZERO;
        }
    }

    {
        let light_pass_params = gpu_pipeline_assets.light_pass_params.get_mut();
        light_pass_params.frame_counter = *gpu_frame_counter;
        light_pass_params.probe_size = GI_SCREEN_PROBE_SIZE;
        light_pass_params.probe_atlas_cols            = gpu_target_sizes.probe_grid_isize.x;
        light_pass_params.probe_atlas_rows            = gpu_target_sizes.probe_grid_isize.y;
        light_pass_params.reservoir_size              = light_pass_config.reservoir_size;
        light_pass_params.smooth_kernel_size_h        = light_pass_config.smooth_kernel_size.0;
        light_pass_params.smooth_kernel_size_w        = light_pass_config.smooth_kernel_size.1;
        light_pass_params.direct_light_contrib        = light_pass_config.direct_light_contrib;
        light_pass_params.indirect_light_contrib      = light_pass_config.indirect_light_contrib;
        light_pass_params.indirect_rays_radius_factor = light_pass_config.indirect_rays_radius_factor;
        light_pass_params.indirect_rays_per_sample    = light_pass_config.indirect_rays_per_sample;
        light_pass_params.boundary_cells              = light_pass_config.boundary_cells;
    }

    {
        let light_pass_params = gpu_pipeline_assets.light_pass_params.get_mut();
        light_pass_params.skylight_color = Vec3::splat(0.0);
        for new_gi_state in query_skylight_light.iter() {
            // Linear, as the light math is.
            let linear = new_gi_state.color.to_linear();
            light_pass_params.skylight_color.x += linear.red * new_gi_state.intensity;
            light_pass_params.skylight_color.y += linear.green * new_gi_state.intensity;
            light_pass_params.skylight_color.z += linear.blue * new_gi_state.intensity;
        }
    }

    *gpu_frame_counter = (*gpu_frame_counter + 1) % (GI_SCREEN_PROBE_SIZE * GI_SCREEN_PROBE_SIZE);
}

/// 0 at or below `edge0`, 1 at or above `edge1`, eased between, as WGSL's.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32
{
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
