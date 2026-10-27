use bevy::asset::uuid_handle;
use bevy::prelude::*;

use crate::gi::compositing::PostProcessingMaterial;

pub const GI_SCREEN_PROBE_SIZE: i32 = 4;

/// How far the SDF reaches past the view, as a multiple of the view's size:
/// 2 covers twice the view's width and height, centered on it.
pub const SDF_VIEW_SCALE: f32 = 2.0;

/// How far the coarse SDF reaches past the view, in the same measure as
/// [`SDF_VIEW_SCALE`]. A march toward a light that leaves the fine SDF goes
/// on in the coarse one, so a light up to this far off screen still lights
/// it, and the walls around it still shade it.
pub const COARSE_SDF_VIEW_SCALE: f32 = 6.0;

/// Where a light starts to fade out on its way to the coarse SDF's edge, as a
/// fraction of the way from the view's center to that edge. A light past the
/// edge lights nothing, so its march never leaves the coarse SDF and never
/// passes walls that SDF cannot see. The fade makes it come and go smoothly.
pub const COARSE_SDF_LIGHT_FADE: f32 = 4.0 / 6.0;

/// How far past an SDF's area an occluder is still uploaded, as a fraction
/// of that area's half size.
pub const SDF_CULL_MARGIN: f32 = 0.25;

pub const POST_PROCESSING_RECT: Handle<Mesh> = uuid_handle!("00000000-0000-0001-45ca-30fd30a6002b");
pub const POST_PROCESSING_MATERIAL: Handle<PostProcessingMaterial> =
    uuid_handle!("00000000-0000-0002-d6d5-ffc73b79ef27");
