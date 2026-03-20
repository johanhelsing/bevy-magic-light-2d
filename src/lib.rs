use bevy::prelude::*;

pub mod gi;
pub mod prelude;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct SpriteCamera;
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct FloorCamera;
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct WallsCamera;
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct ObjectsCamera;
