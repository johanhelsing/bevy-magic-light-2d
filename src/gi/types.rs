use bevy::camera::visibility::{self, VisibilityClass};
use bevy::prelude::*;

#[rustfmt::skip]
#[derive(Reflect, Component, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<OmniLightSource2D>)]
#[reflect(Component, Default)]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct OmniLightSource2D {
    pub intensity:          f32,
    pub color:              Color,
    pub falloff:            Vec3,
    pub jitter_intensity:   f32,
    pub jitter_translation: f32,
    /// Cone (spotlight) shaping. The light only illuminates within a cone of
    /// half-angle `cone_half_angle` around `cone_direction`. The default is a
    /// full circle (`cone_half_angle = PI`), so an unconfigured light is a
    /// plain omnidirectional point light and every existing call site is
    /// unaffected. `cone_direction` need not be normalized; the shader
    /// normalizes it. A soft edge fades over `cone_softness` radians inside the
    /// rim. See cargo-space ADR 0003.
    ///
    /// The cone fields are `#[reflect(default)]` so scenes serialized before
    /// they existed still deserialize — a missing `cone_half_angle` fills with
    /// PI (full circle) via [`default_cone_half_angle`], the others with their
    /// zero value (harmless once the half-angle is full-circle).
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(default))]
    pub cone_direction:     Vec2,
    #[reflect(default = "default_cone_half_angle")]
    #[cfg_attr(feature = "serde", serde(default = "default_cone_half_angle"))]
    pub cone_half_angle:    f32,
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(default))]
    pub cone_softness:      f32,
}

/// Default `cone_half_angle`: a full circle, so an unconfigured or pre-cone
/// light is a plain omnidirectional point light.
pub fn default_cone_half_angle() -> f32
{
    core::f32::consts::PI
}

#[rustfmt::skip]
#[derive(Reflect, Component, Default, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<LightOccluder2D>)]
#[reflect(Component)]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct LightOccluder2D {
    pub h_size: Vec2,
}

impl Default for OmniLightSource2D
{
    fn default() -> Self
    {
        Self {
            intensity:          0.0,
            color:              Color::default(),
            falloff:            Vec3::ZERO,
            jitter_intensity:   0.0,
            jitter_translation: 0.0,
            // Full-circle cone: an unconfigured light is a plain omni point
            // light, so every existing call site and serialized scene is
            // unaffected.
            cone_direction:     Vec2::X,
            cone_half_angle:    core::f32::consts::PI,
            cone_softness:      0.0,
        }
    }
}

impl From<(f32, f32)> for LightOccluder2D
{
    fn from(value: (f32, f32)) -> Self
    {
        LightOccluder2D {
            h_size: value.into(),
        }
    }
}

impl From<Vec2> for LightOccluder2D
{
    fn from(value: Vec2) -> Self
    {
        LightOccluder2D { h_size: value }
    }
}

#[rustfmt::skip]
#[derive(Reflect, Component, Default, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[reflect(Component)]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct SkylightMask2D {
    pub h_size: Vec2,
}

#[rustfmt::skip]
#[derive(Reflect, Component, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[reflect(Component)]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct SkylightLight2D {
    pub color:     Color,
    pub intensity: f32,
}
