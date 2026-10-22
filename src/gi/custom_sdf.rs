//! Occluders of the app's own shape.
//!
//! A [`LightOccluder2D`] with an `sdf` names a shader module that defines
//! `fn sdf(p: vec2<f32>, a: vec4<f32>, b: vec4<f32>) -> f32`, the signed
//! distance to its shape at `p` (meters from the occluder's center, in its
//! frame), with the occluder's `params` as `a` and `b`. The SDF pass calls
//! it inside a disc of the occluder's `h_size.x` and uses the disc's own
//! distance outside, a lower bound the march can step by.
//!
//! The pass's shader imports one module, [`MODULE_PATH`], which this file
//! writes: an import of every such shader in the world and a `switch` over
//! a number each got here. The module is one asset, rewritten in place when
//! the set changes, so the pass recompiles by the ordinary shader reload and
//! the app never names a kind.

use std::collections::HashSet;

use bevy::prelude::*;
use bevy::shader::ShaderImport;

use crate::gi::types::LightOccluder2D;
use crate::prelude::BevyMagicLight2DSettings;

/// The import path of the composed module.
pub const MODULE_PATH: &str = "bevy_magic_light_2d::custom_sdf";

/// The composed module and the number each shape shader has in it.
#[derive(Resource)]
pub struct CustomSdfs
{
    module:     Handle<Shader>,
    /// The module's shapes in kind order, from 1: each shader's import path
    /// and id. Only grows unless the settings say to forget unused ones.
    entries:    Vec<(String, AssetId<Shader>)>,
    /// Shaders an app registered ahead of any occluder, so the module holds
    /// them from the start and a shape's first appearance recompiles
    /// nothing. Strong handles, so the assets stay loaded.
    registered: Vec<Handle<Shader>>,
}

impl CustomSdfs
{
    pub(crate) fn new(shaders: &mut Assets<Shader>) -> Self
    {
        let module = shaders.add(module_shader(&[]));
        Self {
            module,
            entries: Vec::new(),
            registered: Vec::new(),
        }
    }

    /// The pass's number for a shape shader, when it is in the module.
    pub fn kind(&self, shader: &Handle<Shader>) -> Option<u32>
    {
        let id = shader.id();
        self.entries
            .iter()
            .position(|(_, entry)| *entry == id)
            .map(|i| i as u32 + 1)
    }

    /// Put a shape shader in the module before any occluder names it, so
    /// the pass compiles with it once at startup instead of again at the
    /// shape's first appearance. [`OccluderShaderAppExt::add_occluder_shader`]
    /// is this at build time.
    pub fn register(&mut self, shader: Handle<Shader>)
    {
        if !self.registered.contains(&shader) {
            self.registered.push(shader);
        }
    }
}

/// [`CustomSdfs::register`] on the app, after [`crate::BevyMagicLight2DPlugin`].
pub trait OccluderShaderAppExt
{
    fn add_occluder_shader(&mut self, shader: Handle<Shader>) -> &mut Self;
}

impl OccluderShaderAppExt for App
{
    fn add_occluder_shader(&mut self, shader: Handle<Shader>) -> &mut Self
    {
        self.world_mut()
            .get_resource_mut::<CustomSdfs>()
            .expect("add_occluder_shader after BevyMagicLight2DPlugin")
            .register(shader);
        self
    }
}

fn module_shader(imports: &[String]) -> Shader
{
    Shader::from_wgsl(compose(imports), format!("{MODULE_PATH}.wgsl"))
}

/// The module's source: one import and one arm per shape shader.
fn compose(imports: &[String]) -> String
{
    let mut source = format!("#define_import_path {MODULE_PATH}\n");
    for (i, path) in imports.iter().enumerate() {
        source += &format!("#import {path} as sdf_{}\n", i + 1);
    }
    source += "\nfn custom_sdf(p: vec2<f32>, kind: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {\n";
    source += "    switch kind {\n";
    for i in 1..=imports.len() {
        source += &format!("        case {i}u: {{ return sdf_{i}::sdf(p, a, b); }}\n");
    }
    source += "        default: { return 1e10; }\n    }\n}\n";
    source
}

/// Recomposes the module when a shape shader joins: one an occluder in the
/// world names, or one registered, as soon as it is loaded. A shader without
/// a `#define_import_path` is left out with a warning. A shader leaves only
/// when the settings say to forget unused ones and no occluder names it.
pub(crate) fn compose_custom_sdfs(
    occluders: Query<&LightOccluder2D>,
    changed: Query<(), Changed<LightOccluder2D>>,
    mut removed: RemovedComponents<LightOccluder2D>,
    mut shader_events: MessageReader<AssetEvent<Shader>>,
    settings: Res<BevyMagicLight2DSettings>,
    mut shaders: ResMut<Assets<Shader>>,
    mut sdfs: ResMut<CustomSdfs>,
)
{
    let shaders_moved = shader_events.read().count() > 0;
    let registered_moved = sdfs.is_changed();
    if changed.is_empty() && removed.read().next().is_none() && !shaders_moved && !registered_moved
    {
        return;
    }
    let in_use: HashSet<AssetId<Shader>> = occluders
        .iter()
        .filter_map(|o| o.sdf.as_ref())
        .map(Handle::id)
        .collect();
    let mut entries: Vec<(String, AssetId<Shader>)> = if settings.forget_unused_occluder_shaders {
        sdfs.entries
            .iter()
            .filter(|(_, id)| in_use.contains(id))
            .cloned()
            .collect()
    } else {
        sdfs.entries.clone()
    };
    let wanted = in_use
        .iter()
        .copied()
        .chain(sdfs.registered.iter().map(Handle::id));
    for id in wanted {
        if entries.iter().any(|(_, entry)| *entry == id) {
            continue;
        }
        let Some(shader) = shaders.get(id) else {
            continue;
        };
        match &shader.import_path {
            ShaderImport::Custom(path) => entries.push((path.clone(), id)),
            _ => warn!("occluder shader {id:?} has no #define_import_path; it casts nothing"),
        }
    }
    if entries == sdfs.entries {
        return;
    }
    let imports: Vec<String> = entries.iter().map(|(path, _)| path.clone()).collect();
    sdfs.entries = entries;
    let module = sdfs.module.id();
    if let Err(err) = shaders.insert(module, module_shader(&imports)) {
        warn!("the custom occluder module could not be rewritten: {err}");
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn the_module_imports_each_shape_and_dispatches_on_its_number()
    {
        let source = compose(&["game::rock".to_string(), "game::star".to_string()]);
        assert!(source.starts_with("#define_import_path bevy_magic_light_2d::custom_sdf\n"));
        assert!(source.contains("#import game::rock as sdf_1\n"));
        assert!(source.contains("#import game::star as sdf_2\n"));
        assert!(source.contains("case 1u: { return sdf_1::sdf(p, a, b); }"));
        assert!(source.contains("case 2u: { return sdf_2::sdf(p, a, b); }"));
        assert!(source.contains("default: { return 1e10; }"));
    }

    #[test]
    fn an_empty_module_still_defines_the_function()
    {
        let source = compose(&[]);
        assert!(!source.contains("#import"));
        assert!(source
            .contains("fn custom_sdf(p: vec2<f32>, kind: u32, a: vec4<f32>, b: vec4<f32>) -> f32"));
    }
}
