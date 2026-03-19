use bevy::asset::uuid_handle;
use bevy::prelude::*;

use crate::gi::compositing::PostProcessingMaterial;

pub const GI_SCREEN_PROBE_SIZE: i32 = 8;

pub const POST_PROCESSING_RECT: Handle<Mesh> = uuid_handle!("00000000-0000-0001-45ca-30fd30a6002b");
pub const POST_PROCESSING_MATERIAL: Handle<PostProcessingMaterial> =
    uuid_handle!("00000000-0000-0002-d6d5-ffc73b79ef27");
