import bevy_sprite_render::mesh2d::view_bindings::view;
import bevy_sprite_render::mesh2d::vertex_output::VertexOutput;

fn coords_to_viewport_uv(position: vec2<f32>, viewport: vec4<f32>) -> vec2<f32> {
    return (position - viewport.xy) / viewport.zw;
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var in_floor_texture:              texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var in_floor_sampler:              sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var in_walls_texture:              texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var in_walls_sampler:              sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var in_objects_texture:            texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(5) var in_objects_sampler:            sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(6) var in_irradiance_texture:         texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(7) var in_irradiance_texture_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let position = in.position;
    let uv = coords_to_viewport_uv(position.xy, view.viewport);

    // Read diffuse textures.
    let in_floor_diffuse   = textureSample(in_floor_texture,   in_floor_sampler, uv);
    let in_walls_diffuse   = textureSample(in_walls_texture,   in_walls_sampler, uv);
    let in_objects_diffuse = textureSample(in_objects_texture, in_objects_sampler, uv);

    let in_irradiance = textureSample(in_irradiance_texture, in_irradiance_texture_sampler, uv).xyz;

    // Calculate object irradiance.
    // TODO: parametrize this filter.
    // TODO: we don't really need to do this per pixel.
    var object_irradiance = in_irradiance;
    let k_size = 3;
    let k_width = 28;

    for (var i = -k_size; i <= k_size; i++) {
        for (var j = -k_size; j < 0; j++) {

            let offset = vec2<f32>(f32(i * k_width), f32(j * k_width));
            let irradiance_uv = coords_to_viewport_uv(position.xy - offset, view.viewport);

            let sample_irradiance = textureSample(
                in_irradiance_texture,
                in_irradiance_texture_sampler,
                irradiance_uv
            ).xyz;

            // TODO: Might also need a visibility check here.
            if any(irradiance_uv < vec2<f32>(0.0)) || any(irradiance_uv > vec2<f32>(1.0)) {
                continue;
            }

            object_irradiance = max(object_irradiance, sample_irradiance);
        }
    }

    // Irradiance is linear light, 1.0 shows a sprite as drawn. Nothing is
    // encoded here: the camera is HDR and Bevy writes sRGB at the end.
    let final_floor   = in_floor_diffuse.xyz   * in_irradiance;
    let final_walls   = in_walls_diffuse.xyz   * in_irradiance;
    let final_objects = in_objects_diffuse.xyz * object_irradiance;

    // Composite layers. Alpha tracks whether any geometry is present,
    // so areas with no content are transparent (stars show through).
    let has_geometry = max(in_floor_diffuse.a, max(in_walls_diffuse.a, in_objects_diffuse.a));

    var out = vec4<f32>(final_floor, has_geometry);
        out = vec4<f32>(mix(out.xyz, final_walls.xyz, in_walls_diffuse.w), has_geometry);
        out = vec4<f32>(mix(out.xyz, final_objects.xyz, in_objects_diffuse.w), has_geometry);

    return out;
}
