//! Overlay gizmo test — based on the movement example.
//!
//! Renders the same scene as movement (orbiting occluder, two colored lights)
//! with a dedicated overlay camera for gizmos that bypass GI.
//!
//! The overlay camera uses `Hdr` (without `Bloom`) to join the same render
//! target group as the post-processing camera, and `ClearColorConfig::None`
//! to preserve the composited GI scene underneath.
//!
//! Expected: GI-lit scene with green bounding box + yellow crosshair over the
//! occluder, always fully visible regardless of lighting.

use std::f64::consts::PI;

use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::gizmos::config::{DefaultGizmoConfigGroup, GizmoConfigStore};
use bevy::prelude::*;
use bevy::render::view::Hdr;
use bevy::window::WindowResolution;
use bevy_magic_light_2d::prelude::*;

const OVERLAY_LAYER: usize = 50;

#[derive(Debug, Component)]
struct Mover;

#[derive(Component)]
struct OverlayCamera;

fn main()
{
    App::new()
        .insert_resource(ClearColor(Color::srgba_u8(255, 255, 255, 255)))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(1024, 1024),
                    title: "Bevy Magic Light 2D: Overlay Test".into(),
                    resizable: false,
                    ..default()
                }),
                ..default()
            }),
            BevyMagicLight2DPlugin,
        ))
        .register_type::<BevyMagicLight2DSettings>()
        .add_systems(Startup, setup.after(setup_post_processing_camera))
        .add_systems(Startup, configure_gizmo_layers)
        .add_systems(
            Update,
            (system_move_camera, move_collider, draw_overlay_gizmos),
        )
        .insert_resource(BevyMagicLight2DSettings {
            light_pass_params: LightPassParams {
                reservoir_size: 8,
                smooth_kernel_size: (3, 3),
                direct_light_contrib: 0.5,
                indirect_light_contrib: 0.5,
                ..default()
            },
            ..default()
        })
        .run();
}

fn setup(mut commands: Commands, camera_targets: Res<CameraTargets>)
{
    // Occluder (orbits the center)
    let occluder_entity = commands
        .spawn((
            Transform::default(),
            Visibility::default(),
            LightOccluder2D {
                h_size: Vec2::new(80.0, 40.0),
            },
            Mover,
        ))
        .id();

    commands
        .spawn((
            Visibility::default(),
            Transform::default(),
            Name::new("occluders"),
        ))
        .add_children(&[occluder_entity]);

    // Two colored lights in opposite corners
    let light_left = commands
        .spawn((
            Name::new("left"),
            OmniLightSource2D {
                intensity: 10.0,
                color: Color::srgb(1.0, 1.0, 0.0),
                falloff: Vec3::new(1.5, 10.0, 0.01),
                ..default()
            },
            Visibility::default(),
            Transform::from_xyz(-512.0, -512.0, 0.0),
        ))
        .id();

    let light_right = commands
        .spawn((
            Name::new("right"),
            OmniLightSource2D {
                intensity: 10.0,
                color: Color::srgb(0.0, 1.0, 1.0),
                falloff: Vec3::new(1.5, 10.0, 0.01),
                ..default()
            },
            Visibility::default(),
            Transform::from_xyz(512.0, -512.0, 0.0),
        ))
        .id();

    commands
        .spawn((
            Transform::default(),
            Visibility::default(),
            Name::new("lights"),
        ))
        .add_children(&[light_left, light_right]);

    // Floor camera
    commands.spawn((
        Camera2d,
        Camera::default(),
        RenderTarget::Image(camera_targets.floor_target.clone().into()),
        FloorCamera,
        Name::new("floor_camera"),
    ));

    // Overlay camera — renders gizmos on top, bypassing GI.
    // `Hdr` puts it in the same render target group as the post-processing
    // camera so it draws on top rather than overwriting.
    commands.spawn((
        Camera2d,
        Camera {
            order: 2,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        Hdr,
        RenderLayers::layer(OVERLAY_LAYER),
        OverlayCamera,
        Name::new("overlay_camera"),
    ));
}

fn configure_gizmo_layers(mut config_store: ResMut<GizmoConfigStore>)
{
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.render_layers = RenderLayers::layer(OVERLAY_LAYER);
}

fn draw_overlay_gizmos(mut gizmos: Gizmos, movers: Query<&Transform, With<Mover>>)
{
    for transform in &movers {
        let pos = transform.translation.truncate();
        let rot = transform.rotation.to_euler(EulerRot::ZYX).0;
        let isometry = Isometry2d::new(pos, Rot2::radians(rot));
        // Green bounding box + yellow diagonal cross matching occluder rotation
        let half = Vec2::new(82.5, 42.5);
        gizmos.rect_2d(isometry, half * 2.0, Color::srgb(0.0, 1.0, 0.0));
        let right = Vec2::from_angle(rot);
        let up = right.perp();
        let corner_a = pos + right * half.x + up * half.y;
        let corner_b = pos - right * half.x - up * half.y;
        let corner_c = pos + right * half.x - up * half.y;
        let corner_d = pos - right * half.x + up * half.y;
        gizmos.line_2d(corner_a, corner_b, Color::srgb(1.0, 1.0, 0.0));
        gizmos.line_2d(corner_c, corner_d, Color::srgb(1.0, 1.0, 0.0));
    }
}

fn system_move_camera(
    mut camera_target: Local<Vec3>,
    mut query_floor: Query<&mut Transform, (With<FloorCamera>, Without<OverlayCamera>)>,
    mut query_overlay: Query<&mut Transform, (With<OverlayCamera>, Without<FloorCamera>)>,
    keyboard: Res<ButtonInput<KeyCode>>,
)
{
    if let Ok(mut camera_transform) = query_floor.single_mut() {
        let speed = 10.0;
        if keyboard.pressed(KeyCode::KeyW) {
            camera_target.y += speed;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            camera_target.y -= speed;
        }
        if keyboard.pressed(KeyCode::KeyA) {
            camera_target.x -= speed;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            camera_target.x += speed;
        }

        let blend_ratio = 0.18;
        let movement = (*camera_target - camera_transform.translation) * blend_ratio;
        camera_transform.translation.x += movement.x;
        camera_transform.translation.y += movement.y;

        if let Ok(mut overlay_transform) = query_overlay.single_mut() {
            overlay_transform.translation = camera_transform.translation;
        }
    }
}

fn move_collider(mut query_mover: Query<&mut Transform, With<Mover>>, time: Res<Time>)
{
    let radius = 100.;
    let cycle_secs = 5.;
    let elapsed = time.elapsed().as_secs_f64();
    let theta = (elapsed % cycle_secs / cycle_secs) * 2. * PI;

    if let Ok(mut transform) = query_mover.single_mut() {
        transform.translation.x = radius * theta.cos() as f32;
        transform.translation.y = radius * theta.sin() as f32;
        transform.rotation = Quat::from_rotation_z(theta as f32);
    }
}
