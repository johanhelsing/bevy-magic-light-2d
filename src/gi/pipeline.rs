use bevy::asset::{uuid_handle, RenderAssetUsages};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::GpuImage;

use crate::gi::pipeline_assets::{load_embedded_shader, LightPassPipelineAssets};
use crate::gi::resource::ComputedTargetSizes;
use crate::gi::types_gpu::{
    GpuCameraParams,
    GpuLightOccluderBuffer,
    GpuLightPassParams,
    GpuLightSourceBuffer,
    GpuProbeDataBuffer,
    GpuSkylightMaskBuffer,
};

const SDF_TARGET_FORMAT: TextureFormat = TextureFormat::R16Float;
const SS_PROBE_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const SS_BOUNCE_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
const SS_BLEND_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
const SS_FILTER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
const SS_POSE_TARGET_FORMAT: TextureFormat = TextureFormat::Rg32Float;

const SDF_PIPELINE_ENTRY: &str = "main";
const SS_PROBE_PIPELINE_ENTRY: &str = "main";
const SS_BOUNCE_PIPELINE_ENTRY: &str = "main";
const SS_BLEND_PIPELINE_ENTRY: &str = "main";
const SS_FILTER_PIPELINE_ENTRY: &str = "main";

#[allow(dead_code)]
#[derive(Clone, Resource, ExtractResource, Default)]
pub struct GiTargetsWrapper
{
    pub targets: Option<GiTargets>,
}

#[derive(Clone)]
pub struct GiTargets
{
    pub sdf_target:       Handle<Image>,
    pub ss_probe_target:  Handle<Image>,
    pub ss_bounce_target: Handle<Image>,
    pub ss_blend_target:  Handle<Image>,
    pub ss_filter_target: Handle<Image>,
    pub ss_pose_target:   Handle<Image>,
}

impl GiTargets
{
    pub fn create(images: &mut Assets<Image>, sizes: &ComputedTargetSizes) -> Self
    {
        let sdf_tex = create_texture_2d(
            sizes.sdf_target_usize.into(),
            SDF_TARGET_FORMAT,
            ImageFilterMode::Linear,
        );
        let ss_probe_tex = create_texture_2d(
            sizes.primary_target_usize.into(),
            SS_PROBE_TARGET_FORMAT,
            ImageFilterMode::Nearest,
        );
        let ss_bounce_tex = create_texture_2d(
            sizes.primary_target_usize.into(),
            SS_BOUNCE_TARGET_FORMAT,
            ImageFilterMode::Nearest,
        );
        let ss_blend_tex = create_texture_2d(
            sizes.probe_grid_usize.into(),
            SS_BLEND_TARGET_FORMAT,
            ImageFilterMode::Nearest,
        );
        let ss_filter_tex = create_texture_2d(
            sizes.primary_target_usize.into(),
            SS_FILTER_TARGET_FORMAT,
            ImageFilterMode::Nearest,
        );
        let ss_pose_tex = create_texture_2d(
            sizes.primary_target_usize.into(),
            SS_POSE_TARGET_FORMAT,
            ImageFilterMode::Nearest,
        );

        let sdf_target = uuid_handle!("00000000-0000-0000-212d-fedea82bb2d7");
        let ss_probe_target = uuid_handle!("00000000-0000-0000-2f81-c2ac3e832cda");
        let ss_bounce_target = uuid_handle!("00000000-0000-0000-2c62-8c87580fa5a7");
        let ss_blend_target = uuid_handle!("00000000-0000-0000-6c00-54342dfbf209");
        let ss_filter_target = uuid_handle!("00000000-0000-0000-7996-26a4fdeb17e4");
        let ss_pose_target = uuid_handle!("00000000-0000-0000-419d-d117fc4b32d6");

        images
            .insert(sdf_target.id(), sdf_tex)
            .expect("failed to insert image");
        images
            .insert(ss_probe_target.id(), ss_probe_tex)
            .expect("failed to insert image");
        images
            .insert(ss_bounce_target.id(), ss_bounce_tex)
            .expect("failed to insert image");
        images
            .insert(ss_blend_target.id(), ss_blend_tex)
            .expect("failed to insert image");
        images
            .insert(ss_filter_target.id(), ss_filter_tex)
            .expect("failed to insert image");
        images
            .insert(ss_pose_target.id(), ss_pose_tex)
            .expect("failed to insert image");

        Self {
            sdf_target,
            ss_probe_target,
            ss_bounce_target,
            ss_blend_target,
            ss_filter_target,
            ss_pose_target,
        }
    }
}

#[allow(dead_code)]
#[derive(Resource)]
pub struct LightPassPipelineBindGroups
{
    pub sdf_bind_group:       BindGroup,
    pub ss_blend_bind_group:  BindGroup,
    pub ss_probe_bind_group:  BindGroup,
    pub ss_bounce_bind_group: BindGroup,
    pub ss_filter_bind_group: BindGroup,
}

#[rustfmt::skip]
fn create_texture_2d(size: (u32, u32), format: TextureFormat, filter: ImageFilterMode) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.0,
            height: size.1,
            ..Default::default()
        },
        TextureDimension::D2,
        &[
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        format,
        RenderAssetUsages::default(),
    );

    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: filter,
        min_filter: filter,
        address_mode_u: ImageAddressMode::ClampToBorder,
        address_mode_v: ImageAddressMode::ClampToBorder,
        address_mode_w: ImageAddressMode::ClampToBorder,
        ..Default::default()
    });

    image
}

#[rustfmt::skip]
pub fn system_setup_gi_pipeline(
    mut images:          ResMut<Assets<Image>>,
    mut targets_wrapper: ResMut<GiTargetsWrapper>,
    targets_sizes:   Res<ComputedTargetSizes>,
) {
    targets_wrapper.targets = Some(GiTargets::create(&mut images, &targets_sizes));
}

#[derive(Resource)]
pub struct LightPassPipeline
{
    pub sdf_bind_group_layout:       BindGroupLayoutDescriptor,
    pub sdf_pipeline:                CachedComputePipelineId,
    pub ss_probe_bind_group_layout:  BindGroupLayoutDescriptor,
    pub ss_probe_pipeline:           CachedComputePipelineId,
    pub ss_bounce_bind_group_layout: BindGroupLayoutDescriptor,
    pub ss_bounce_pipeline:          CachedComputePipelineId,
    pub ss_blend_bind_group_layout:  BindGroupLayoutDescriptor,
    pub ss_blend_pipeline:           CachedComputePipelineId,
    pub ss_filter_bind_group_layout: BindGroupLayoutDescriptor,
    pub ss_filter_pipeline:          CachedComputePipelineId,
}

pub fn system_queue_bind_groups(
    mut commands: Commands,
    pipeline: Res<LightPassPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    targets_wrapper: Res<GiTargetsWrapper>,
    gi_compute_assets: Res<LightPassPipelineAssets>,
    render_device: Res<RenderDevice>,
)
{
    let light_sources_binding = gi_compute_assets.light_sources.binding();
    let light_occluders_binding = gi_compute_assets.light_occluders.binding();
    let camera_params_binding = gi_compute_assets.camera_params.binding();
    let gi_state_binding = gi_compute_assets.light_pass_params.binding();
    let probes_binding = gi_compute_assets.probes.binding();
    let skylight_masks_binding = gi_compute_assets.skylight_masks.binding();

    if light_sources_binding.is_none()
        || light_occluders_binding.is_none()
        || camera_params_binding.is_none()
        || gi_state_binding.is_none()
        || probes_binding.is_none()
        || skylight_masks_binding.is_none()
    {
        // Expected on the first frame before extract/prepare have run
        return;
    }

    if let (
        Some(light_sources),
        Some(light_occluders),
        Some(camera_params),
        Some(gi_state),
        Some(probes),
        Some(skylight_masks),
    ) = (
        light_sources_binding,
        light_occluders_binding,
        camera_params_binding,
        gi_state_binding,
        probes_binding,
        skylight_masks_binding,
    ) {
        let targets = targets_wrapper
            .targets
            .as_ref()
            .expect("Targets should be initialized");

        // GPU images may not be ready yet after target recreation (resize).
        // Return without creating bind groups — the render node will skip this frame.
        let (
            Some(sdf_view_image),
            Some(ss_probe_image),
            Some(ss_bounce_image),
            Some(ss_blend_image),
            Some(ss_filter_image),
            Some(ss_pose_image),
        ) = (
            gpu_images.get(&targets.sdf_target),
            gpu_images.get(&targets.ss_probe_target),
            gpu_images.get(&targets.ss_bounce_target),
            gpu_images.get(&targets.ss_blend_target),
            gpu_images.get(&targets.ss_filter_target),
            gpu_images.get(&targets.ss_pose_target),
        )
        else {
            // Clear stale bind groups so the render node doesn't use old GPU textures.
            commands.remove_resource::<LightPassPipelineBindGroups>();
            return;
        };

        let sdf_layout = pipeline_cache.get_bind_group_layout(&pipeline.sdf_bind_group_layout);
        let ss_probe_layout =
            pipeline_cache.get_bind_group_layout(&pipeline.ss_probe_bind_group_layout);
        let ss_bounce_layout =
            pipeline_cache.get_bind_group_layout(&pipeline.ss_bounce_bind_group_layout);
        let ss_blend_layout =
            pipeline_cache.get_bind_group_layout(&pipeline.ss_blend_bind_group_layout);
        let ss_filter_layout =
            pipeline_cache.get_bind_group_layout(&pipeline.ss_filter_bind_group_layout);

        let sdf_bind_group = render_device.create_bind_group(
            "gi_sdf_bind_group",
            &sdf_layout,
            &[
                BindGroupEntry {
                    binding:  0,
                    resource: camera_params.clone(),
                },
                BindGroupEntry {
                    binding:  1,
                    resource: light_occluders.clone(),
                },
                BindGroupEntry {
                    binding:  2,
                    resource: BindingResource::TextureView(&sdf_view_image.texture_view),
                },
            ],
        );

        let ss_probe_bind_group = render_device.create_bind_group(
            "gi_ss_probe_bind_group",
            &ss_probe_layout,
            &[
                BindGroupEntry {
                    binding:  0,
                    resource: camera_params.clone(),
                },
                BindGroupEntry {
                    binding:  1,
                    resource: gi_state.clone(),
                },
                BindGroupEntry {
                    binding:  2,
                    resource: probes.clone(),
                },
                BindGroupEntry {
                    binding:  3,
                    resource: skylight_masks.clone(),
                },
                BindGroupEntry {
                    binding:  4,
                    resource: light_sources.clone(),
                },
                BindGroupEntry {
                    binding:  5,
                    resource: BindingResource::TextureView(&sdf_view_image.texture_view),
                },
                BindGroupEntry {
                    binding:  6,
                    resource: BindingResource::Sampler(&sdf_view_image.sampler),
                },
                BindGroupEntry {
                    binding:  7,
                    resource: BindingResource::TextureView(&ss_probe_image.texture_view),
                },
            ],
        );

        let ss_bounce_bind_group = render_device.create_bind_group(
            "gi_bounce_bind_group",
            &ss_bounce_layout,
            &[
                BindGroupEntry {
                    binding:  0,
                    resource: camera_params.clone(),
                },
                BindGroupEntry {
                    binding:  1,
                    resource: gi_state.clone(),
                },
                BindGroupEntry {
                    binding:  2,
                    resource: BindingResource::TextureView(&sdf_view_image.texture_view),
                },
                BindGroupEntry {
                    binding:  3,
                    resource: BindingResource::Sampler(&sdf_view_image.sampler),
                },
                BindGroupEntry {
                    binding:  4,
                    resource: BindingResource::TextureView(&ss_probe_image.texture_view),
                },
                BindGroupEntry {
                    binding:  5,
                    resource: BindingResource::TextureView(&ss_bounce_image.texture_view),
                },
            ],
        );

        let ss_blend_bind_group = render_device.create_bind_group(
            "gi_blend_bind_group",
            &ss_blend_layout,
            &[
                BindGroupEntry {
                    binding:  0,
                    resource: camera_params.clone(),
                },
                BindGroupEntry {
                    binding:  1,
                    resource: gi_state.clone(),
                },
                BindGroupEntry {
                    binding:  2,
                    resource: probes.clone(),
                },
                BindGroupEntry {
                    binding:  3,
                    resource: BindingResource::TextureView(&sdf_view_image.texture_view),
                },
                BindGroupEntry {
                    binding:  4,
                    resource: BindingResource::Sampler(&sdf_view_image.sampler),
                },
                BindGroupEntry {
                    binding:  5,
                    resource: BindingResource::TextureView(&ss_bounce_image.texture_view),
                },
                BindGroupEntry {
                    binding:  6,
                    resource: BindingResource::TextureView(&ss_blend_image.texture_view),
                },
            ],
        );

        let ss_filter_bind_group = render_device.create_bind_group(
            "ss_filter_bind_group",
            &ss_filter_layout,
            &[
                BindGroupEntry {
                    binding:  0,
                    resource: camera_params.clone(),
                },
                BindGroupEntry {
                    binding:  1,
                    resource: gi_state.clone(),
                },
                BindGroupEntry {
                    binding:  2,
                    resource: probes.clone(),
                },
                BindGroupEntry {
                    binding:  3,
                    resource: BindingResource::TextureView(&sdf_view_image.texture_view),
                },
                BindGroupEntry {
                    binding:  4,
                    resource: BindingResource::Sampler(&sdf_view_image.sampler),
                },
                BindGroupEntry {
                    binding:  5,
                    resource: BindingResource::TextureView(&ss_blend_image.texture_view),
                },
                BindGroupEntry {
                    binding:  6,
                    resource: BindingResource::TextureView(&ss_filter_image.texture_view),
                },
                BindGroupEntry {
                    binding:  7,
                    resource: BindingResource::TextureView(&ss_pose_image.texture_view),
                },
            ],
        );

        commands.insert_resource(LightPassPipelineBindGroups {
            sdf_bind_group,
            ss_probe_bind_group,
            ss_bounce_bind_group,
            ss_blend_bind_group,
            ss_filter_bind_group,
        });
    }
}

impl FromWorld for LightPassPipeline
{
    fn from_world(world: &mut World) -> Self
    {
        let sdf_bind_group_layout = BindGroupLayoutDescriptor::new(
            "sdf_bind_group_layout",
            &[
                // Camera.
                BindGroupLayoutEntry {
                    binding:    0,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuCameraParams::min_size()),
                    },
                    count:      None,
                },
                // Light occluders.
                BindGroupLayoutEntry {
                    binding:    1,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuLightOccluderBuffer::min_size()),
                    },
                    count:      None,
                },
                // SDF texture.
                BindGroupLayoutEntry {
                    binding:    2,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::ReadWrite,
                        format:         SDF_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
            ],
        );

        let ss_probe_bind_group_layout = BindGroupLayoutDescriptor::new(
            "ss_probe_bind_group_layout",
            &[
                // Camera.
                BindGroupLayoutEntry {
                    binding:    0,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuCameraParams::min_size()),
                    },
                    count:      None,
                },
                // GI State.
                BindGroupLayoutEntry {
                    binding:    1,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuLightPassParams::min_size()),
                    },
                    count:      None,
                },
                // Probes.
                BindGroupLayoutEntry {
                    binding:    2,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuProbeDataBuffer::min_size()),
                    },
                    count:      None,
                },
                // SkylightMasks.
                BindGroupLayoutEntry {
                    binding:    3,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuSkylightMaskBuffer::min_size()),
                    },
                    count:      None,
                },
                // Light sources.
                BindGroupLayoutEntry {
                    binding:    4,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuLightSourceBuffer::min_size()),
                    },
                    count:      None,
                },
                // SDF.
                BindGroupLayoutEntry {
                    binding:    5,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Texture {
                        sample_type:    TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count:      None,
                },
                // SDF Sampler.
                BindGroupLayoutEntry {
                    binding:    6,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Sampler(SamplerBindingType::Filtering),
                    count:      None,
                },
                // SS Probe.
                BindGroupLayoutEntry {
                    binding:    7,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::WriteOnly,
                        format:         SS_PROBE_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
            ],
        );

        let ss_bounce_bind_group_layout = BindGroupLayoutDescriptor::new(
            "ss_bounce_bind_group_layout",
            &[
                // Camera.
                BindGroupLayoutEntry {
                    binding:    0,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuCameraParams::min_size()),
                    },
                    count:      None,
                },
                // GI State.
                BindGroupLayoutEntry {
                    binding:    1,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuLightPassParams::min_size()),
                    },
                    count:      None,
                },
                // SDF.
                BindGroupLayoutEntry {
                    binding:    2,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Texture {
                        sample_type:    TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count:      None,
                },
                // SDF Sampler.
                BindGroupLayoutEntry {
                    binding:    3,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Sampler(SamplerBindingType::Filtering),
                    count:      None,
                },
                // SS Probe.
                BindGroupLayoutEntry {
                    binding:    4,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::ReadOnly,
                        format:         SS_PROBE_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
                // SS Bounce.
                BindGroupLayoutEntry {
                    binding:    5,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::WriteOnly,
                        format:         SS_BOUNCE_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
            ],
        );

        let ss_blend_bind_group_layout = BindGroupLayoutDescriptor::new(
            "ss_blend_bind_group_layout",
            &[
                // Camera.
                BindGroupLayoutEntry {
                    binding:    0,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuCameraParams::min_size()),
                    },
                    count:      None,
                },
                // GI State.
                BindGroupLayoutEntry {
                    binding:    1,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuLightPassParams::min_size()),
                    },
                    count:      None,
                },
                // Probes.
                BindGroupLayoutEntry {
                    binding:    2,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuProbeDataBuffer::min_size()),
                    },
                    count:      None,
                },
                // SDF.
                BindGroupLayoutEntry {
                    binding:    3,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Texture {
                        sample_type:    TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count:      None,
                },
                // SDF Sampler.
                BindGroupLayoutEntry {
                    binding:    4,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Sampler(SamplerBindingType::Filtering),
                    count:      None,
                },
                // SS Bounces.
                BindGroupLayoutEntry {
                    binding:    5,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::ReadOnly,
                        format:         SS_BOUNCE_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
                // SS Blend.
                BindGroupLayoutEntry {
                    binding:    6,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::WriteOnly,
                        format:         SS_BLEND_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
            ],
        );

        let ss_filter_bind_group_layout = BindGroupLayoutDescriptor::new(
            "ss_filter_bind_group_layout",
            &[
                // Camera.
                BindGroupLayoutEntry {
                    binding:    0,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuCameraParams::min_size()),
                    },
                    count:      None,
                },
                // GI State.
                BindGroupLayoutEntry {
                    binding:    1,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuLightPassParams::min_size()),
                    },
                    count:      None,
                },
                // Probes.
                BindGroupLayoutEntry {
                    binding:    2,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Buffer {
                        ty:                 BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(GpuProbeDataBuffer::min_size()),
                    },
                    count:      None,
                },
                // SDF.
                BindGroupLayoutEntry {
                    binding:    3,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Texture {
                        sample_type:    TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count:      None,
                },
                // SDF Sampler.
                BindGroupLayoutEntry {
                    binding:    4,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::Sampler(SamplerBindingType::Filtering),
                    count:      None,
                },
                // SS Blend.
                BindGroupLayoutEntry {
                    binding:    5,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::ReadOnly,
                        format:         SS_BLEND_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
                // SS Filter.
                BindGroupLayoutEntry {
                    binding:    6,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::WriteOnly,
                        format:         SS_FILTER_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
                // SS pose.
                BindGroupLayoutEntry {
                    binding:    7,
                    visibility: ShaderStages::COMPUTE,
                    ty:         BindingType::StorageTexture {
                        access:         StorageTextureAccess::WriteOnly,
                        format:         SS_POSE_TARGET_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count:      None,
                },
            ],
        );

        let (shader_sdf, gi_ss_probe, gi_ss_bounce, gi_ss_blend, gi_ss_filter) = {
            let assets_server = world.resource::<AssetServer>();
            (
                load_embedded_shader(assets_server, "gi_sdf.wgsl"),
                load_embedded_shader(assets_server, "gi_ss_probe.wgsl"),
                load_embedded_shader(assets_server, "gi_ss_bounce.wgsl"),
                load_embedded_shader(assets_server, "gi_ss_blend.wgsl"),
                load_embedded_shader(assets_server, "gi_ss_filter.wgsl"),
            )
        };

        let pipeline_cache = world.resource_mut::<PipelineCache>();

        let sdf_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                            Some("gi_sdf_pipeline".into()),
            layout:                           vec![sdf_bind_group_layout.clone()],
            shader:                           shader_sdf,
            shader_defs:                      vec![],
            entry_point:                      Some(SDF_PIPELINE_ENTRY.into()),
            push_constant_ranges:             vec![],
            zero_initialize_workgroup_memory: false,
        });

        let ss_probe_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                            Some("gi_ss_probe_pipeline".into()),
            layout:                           vec![ss_probe_bind_group_layout.clone()],
            shader:                           gi_ss_probe,
            shader_defs:                      vec![],
            entry_point:                      Some(SS_PROBE_PIPELINE_ENTRY.into()),
            push_constant_ranges:             vec![],
            zero_initialize_workgroup_memory: false,
        });

        let ss_bounce_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                            Some("gi_ss_bounce_pipeline".into()),
            layout:                           vec![ss_bounce_bind_group_layout.clone()],
            shader:                           gi_ss_bounce,
            shader_defs:                      vec![],
            entry_point:                      Some(SS_BOUNCE_PIPELINE_ENTRY.into()),
            push_constant_ranges:             vec![],
            zero_initialize_workgroup_memory: false,
        });

        let ss_blend_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                            Some("gi_blend_pipeline".into()),
            layout:                           vec![ss_blend_bind_group_layout.clone()],
            shader:                           gi_ss_blend,
            shader_defs:                      vec![],
            entry_point:                      Some(SS_BLEND_PIPELINE_ENTRY.into()),
            push_constant_ranges:             vec![],
            zero_initialize_workgroup_memory: false,
        });

        let ss_filter_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label:                            Some("gi_filer_pipeline".into()),
            layout:                           vec![ss_filter_bind_group_layout.clone()],
            shader:                           gi_ss_filter,
            shader_defs:                      vec![],
            entry_point:                      Some(SS_FILTER_PIPELINE_ENTRY.into()),
            push_constant_ranges:             vec![],
            zero_initialize_workgroup_memory: false,
        });

        LightPassPipeline {
            //
            sdf_bind_group_layout,
            sdf_pipeline,
            //
            ss_probe_bind_group_layout,
            ss_probe_pipeline,
            //
            ss_bounce_bind_group_layout,
            ss_bounce_pipeline,
            //
            ss_blend_bind_group_layout,
            ss_blend_pipeline,
            //
            ss_filter_bind_group_layout,
            ss_filter_pipeline,
        }
    }
}
