pub use crate::gi::compositing::{setup_post_processing_camera, CameraTargets};
pub use crate::gi::render_layer::{
    CAMERA_LAYER_FLOOR,
    CAMERA_LAYER_OBJECTS,
    CAMERA_LAYER_POST_PROCESSING,
    CAMERA_LAYER_WALLS,
    LAYER_POST_PROCESSING_ID,
};
pub use crate::gi::resource::{BevyMagicLight2DSettings, LightPassParams};
pub use crate::gi::types::{LightOccluder2D, OmniLightSource2D, SkylightLight2D, SkylightMask2D};
pub use crate::gi::BevyMagicLight2DPlugin;
pub use crate::{FloorCamera, ObjectsCamera, SpriteCamera, WallsCamera};
