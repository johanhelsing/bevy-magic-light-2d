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

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use bevy::shader::ShaderImport;

use crate::gi::types::LightOccluder2D;

/// The import path of the composed module.
pub const MODULE_PATH: &str = "bevy_magic_light_2d::custom_sdf";

/// The composed module and the number each shape shader has in it.
#[derive(Resource)]
pub struct CustomSdfs
{
    module:  Handle<Shader>,
    /// By the shader's id: the `kind` the SDF pass dispatches on, from 1.
    kinds:   HashMap<AssetId<Shader>, u32>,
    /// The import paths in kind order: what the module was composed from.
    imports: Vec<String>,
}

impl CustomSdfs
{
    pub(crate) fn new(shaders: &mut Assets<Shader>) -> Self
    {
        let module = shaders.add(module_shader(&[]));
        Self {
            module,
            kinds: HashMap::new(),
            imports: Vec::new(),
        }
    }

    /// The pass's number for a shape shader, when it is in the module.
    pub fn kind(&self, shader: &Handle<Shader>) -> Option<u32>
    {
        self.kinds.get(&shader.id()).copied()
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

/// Recomposes the module when the set of shape shaders in the world
/// changes, or a shader loads: a shader not loaded yet is left out until it
/// is, and one without a `#define_import_path` is left out with a warning.
pub(crate) fn compose_custom_sdfs(
    occluders: Query<&LightOccluder2D>,
    changed: Query<(), Changed<LightOccluder2D>>,
    mut removed: RemovedComponents<LightOccluder2D>,
    mut shader_events: MessageReader<AssetEvent<Shader>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut sdfs: ResMut<CustomSdfs>,
)
{
    let shaders_moved = shader_events.read().count() > 0;
    if changed.is_empty() && removed.read().next().is_none() && !shaders_moved {
        return;
    }
    let ids: HashSet<AssetId<Shader>> = occluders
        .iter()
        .filter_map(|o| o.sdf.as_ref())
        .map(Handle::id)
        .collect();
    let mut named: Vec<(String, AssetId<Shader>)> = ids
        .into_iter()
        .filter_map(|id| {
            let shader = shaders.get(id)?;
            match &shader.import_path {
                ShaderImport::Custom(path) => Some((path.clone(), id)),
                _ => {
                    warn!("occluder shader {id:?} has no #define_import_path; it casts nothing");
                    None
                }
            }
        })
        .collect();
    named.sort();
    let imports: Vec<String> = named.iter().map(|(path, _)| path.clone()).collect();
    if imports == sdfs.imports {
        return;
    }
    sdfs.kinds = named
        .iter()
        .enumerate()
        .map(|(i, (_, id))| (*id, i as u32 + 1))
        .collect();
    sdfs.imports = imports;
    let module = sdfs.module.id();
    if let Err(err) = shaders.insert(module, module_shader(&sdfs.imports)) {
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
