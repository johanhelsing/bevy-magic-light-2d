use bevy::prelude::{Component, Vec2, Color};
use bevy_inspector_egui::Inspectable;
use bevy::prelude::*;


#[derive(Reflect, Component, Inspectable, Clone, Copy, Default)]
#[reflect(Component)]
pub struct LightSource {
    pub radius:    f32,
    pub intensity: f32,
    pub color:     Color,
}

#[derive(Reflect, Component, Default)]
#[reflect(Component)]
pub struct LightOccluder {
    pub h_size: Vec2,
}

#[derive(Reflect, Component, Default)]
#[reflect(Component)]
pub struct DebugLight;

