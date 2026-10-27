use bevy::prelude::*;

#[rustfmt::skip]
#[derive(Reflect, Component, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
// Visibility brings the InheritedVisibility the extraction reads, and
// Transform the GlobalTransform: without them a light or an occluder is
// skipped with no warning. Not the view's frustum: a light or an occluder
// off screen still lights or shades it (`pipeline_assets`).
#[require(Visibility, Transform)]
#[reflect(Component, Default)]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct OmniLightSource2D {
    /// Linear light at the source: 1.0 shows a sprite as drawn, 2.0 twice as
    /// bright. Past 1.0 a white sprite clips to white.
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
    /// Near-field fade radius in world units. The light ramps from 0 at its
    /// center up to full over this distance, muting the jagged region near the
    /// cone apex where each probe spans a large angle. `0` (default) disables it.
    /// Fixed physical size (independent of camera zoom).
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(default))]
    pub cone_near_fade:     f32,
}

/// Default `cone_half_angle`: a full circle, so an unconfigured or pre-cone
/// light is a plain omnidirectional point light.
pub fn default_cone_half_angle() -> f32
{
    core::f32::consts::PI
}

#[rustfmt::skip]
#[derive(Reflect, Component, Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
// Visibility brings the InheritedVisibility the extraction reads, and
// Transform the GlobalTransform: without them a light or an occluder is
// skipped with no warning. Not the view's frustum: a light or an occluder
// off screen still lights or shades it (`pipeline_assets`).
#[require(Visibility, Transform)]
#[reflect(Component)]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct LightOccluder2D {
    /// Half the occluder's bounding box.
    pub h_size: Vec2,
    /// How far its corners are rounded, 0 for a box. Every shape the SDF
    /// pass draws is a rounded box: a circle is a square rounded by half
    /// its side, a capsule a box rounded by half its width, and a shape
    /// shrunk by a margin keeps its outline when the radius shrinks by the
    /// same margin. Absent from a file saved before it existed: 0.
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(default))]
    pub radius: f32,
    /// A shape of the app's own instead of the rounded box: a shader module
    /// with `fn sdf(p: vec2<f32>, a: vec4<f32>, b: vec4<f32>) -> f32`, the
    /// signed distance at `p` meters from the center in the occluder's
    /// frame, `params` as `a` and `b`. Exact inside a disc of `h_size.x`
    /// and the disc's own distance outside it, which is a lower bound the
    /// march can step by; `radius` is then an inset, how far inside the
    /// shape the shadow starts. Not saved: presentation gives it
    /// (`gi::custom_sdf`).
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub sdf: Option<Handle<Shader>>,
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(default))]
    pub params: [f32; 8],
    /// How far light reaches into the occluder from its surface, meters. A
    /// probe inside is lit as the nearest surface is, faded by
    /// `exp(-depth / lit_depth)` and never below `lit_floor`. At 0 the
    /// interior gets the skylight alone, a cave: a wall's default.
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(default))]
    pub lit_depth: f32,
    /// The least share of the surface's light the interior gets, 0 to 1.
    /// 1 lights the whole interior as its surface: a solid, whose shadow
    /// falls only on what is behind it.
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(default))]
    pub lit_floor: f32,
    /// Which way a lamp's light reaches a point inside: false, from the
    /// nearest point of the surface, lit or not, so the side away from the
    /// lamp copies its shadowed surface; true, through the occluder toward
    /// each lamp, faded by the distance it crosses, so a far side fades
    /// instead. Only an interior with depth or floor above 0 is lit at all.
    #[reflect(default)]
    #[cfg_attr(feature = "serde", serde(default))]
    pub lit_through: bool,
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
            cone_near_fade:     0.0,
        }
    }
}

impl From<(f32, f32)> for LightOccluder2D
{
    fn from(value: (f32, f32)) -> Self
    {
        LightOccluder2D {
            h_size: value.into(),
            ..default()
        }
    }
}

impl From<Vec2> for LightOccluder2D
{
    fn from(value: Vec2) -> Self
    {
        LightOccluder2D {
            h_size: value,
            ..default()
        }
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
    /// Linear light everywhere the sky reaches, in the unit of
    /// [`OmniLightSource2D::intensity`].
    pub intensity: f32,
}
