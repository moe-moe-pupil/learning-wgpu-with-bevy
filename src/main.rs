use bevy::{
    core::Pod,
    diagnostic::{FrameTimeDiagnosticsPlugin, DiagnosticsStore},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::encase::StorageBuffer,
        render_resource::{encase::private::WriteInto, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Render, RenderApp, RenderSet,
    },
    utils::HashMap,
    window::{close_on_esc, WindowMode, WindowPlugin},
};
use bytemuck::{cast_slice, Zeroable};
use line_drawing;
use std::{borrow::Cow, println};
use wgpu::MaintainBase;

const SIZE: (f32, f32) = (512.0, 512.0);

const NUM_MATTERS: u32 = (SIZE.0 * SIZE.1) as u32;

/// Converts cursor position to world coordinates.
/// Convert mouse pos to be origo centered. Then scale it with camera scale, lastly offset
/// by camera position.
pub fn cursor_to_world(window: &Window, camera_pos: Vec2, camera_scale: f32) -> Vec2 {
    (window.cursor_position().unwrap() - Vec2::new(window.width() / 2.0, window.height() / 2.0))
        * camera_scale
        - camera_pos
}

#[derive(Resource, Clone, ExtractResource)]
pub struct GlobalStorage {
    buffers: HashMap<String, Buffer>,
    stage_buffers: HashMap<String, StagingBuffer>,
}

#[derive(Clone, Debug)]
pub struct StagingBuffer {
    mapped: bool,
    buffer: Buffer,
}

impl GlobalStorage {
    #[inline]
    fn unmap_staging_buffers(&mut self) -> &mut Self {
        for (_, staging_buffer) in self.stage_buffers.iter_mut() {
            if staging_buffer.mapped {
                staging_buffer.buffer.unmap();
                staging_buffer.mapped = false;
            }
        }
        self
    }

    #[inline]
    fn map_staging_buffers(&mut self) -> &mut Self {
        for (_, staging_buffer) in self.stage_buffers.iter_mut() {
            let read_buffer_slice = staging_buffer.buffer.slice(..);

            read_buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    println!("{:?}", result);
            });

            staging_buffer.mapped = true;
        }
        self
    }

    #[inline]
    fn add_buffer<T: ShaderType + WriteInto>(
        &mut self,
        name: &str,
        storage: &T,
        render_device: &RenderDevice,
    ) {
        let mut buffer = StorageBuffer::new(Vec::new());
        buffer.write::<T>(storage).unwrap();
        let storage_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(name),
            contents: buffer.as_ref(),
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
        });

        let stage_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(name),
            size: storage_buffer.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.stage_buffers.insert(
            name.to_owned(),
            StagingBuffer {
                buffer: stage_buffer,
                mapped: false,
            },
        );
        self.buffers.insert(name.to_owned(), storage_buffer);
    }
}

/// Mouse world position
#[derive(Debug, Copy, Clone)]
pub struct MousePos {
    pub world: Vec2,
}

#[derive(ShaderType, Pod, Zeroable, Clone, Copy, Debug)]
#[repr(C)]
struct Matter {
    pos: Vec2,
    vel: Vec2,
}

impl MousePos {
    pub fn new(pos: Vec2) -> MousePos {
        MousePos { world: pos }
    }

    /// Converts world position to canvas position:
    /// Inverts y and adds half canvas to the position (pixel units)
    pub fn canvas_pos(&self) -> Vec2 {
        self.world + Vec2::new(SIZE.1 as f32 / 2.0, SIZE.1 as f32 / 2.0)
    }
}

/// Gets a line of canvas coordinates between previous and current mouse position
pub fn get_canvas_line(prev: Option<MousePos>, current: MousePos) -> Vec<IVec2> {
    let canvas_pos = current.canvas_pos();
    let prev = if let Some(prev) = prev {
        prev.canvas_pos()
    } else {
        canvas_pos
    };
    line_drawing::Bresenham::new(
        (prev.x.round() as i32, prev.y.round() as i32),
        (canvas_pos.x.round() as i32, canvas_pos.y.round() as i32),
    )
    .into_iter()
    .map(|pos| IVec2::new(pos.0, pos.1))
    .collect::<Vec<IVec2>>()
}

fn main() {
    //env_logger::init();
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy".into(),
                resolution: SIZE.into(),
                resizable: false,
                present_mode: bevy::window::PresentMode::Immediate,
                mode: WindowMode::Windowed,
                ..default()
            }),
            ..default()
        }))
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_systems(Update, close_on_esc)
        .add_systems(Update, text_update_system)
        .add_systems(Startup, setup)
        .add_systems(Update, (unmap_all, copy_buffer, map_all).chain())
        .add_systems(Update, on_click_compute)
        .add_plugin(PixelSimulationComputePlugin)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0 as u32,
            height: SIZE.1 as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image = images.add(image);

    commands.spawn(SpriteBundle {
        sprite: Sprite {
            custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
            ..default()
        },
        texture: image.clone(),
        ..default()
    });
    commands.spawn(Camera2dBundle::default());

    commands.insert_resource(PixelSimulationImage(image));

    let font = asset_server.load::<Font, _>("fonts/NotoSansTC-Medium.otf");
    commands.spawn((
        TextBundle::from_sections([
            TextSection::new(
                "FPS: ".to_string(),
                TextStyle {
                    font: font.clone(),
                    font_size: 40.0,
                    color: Color::WHITE,
                },
            ),
            TextSection::new(
                "FPS: ".to_string(),
                TextStyle {
                    font: font.clone(),
                    font_size: 40.0,
                    color: Color::WHITE,
                },
            ),
        ])
        .with_text_alignment(TextAlignment::Left)
        .with_style(Style {
            position_type: PositionType::Absolute,
            margin: UiRect {
                left: Val::Px(50.0),
                top: Val::Px(50.0),
                ..default()
            },
            ..default()
        }),
        FpsText,
    ));

    let mut initial_matter_data = Vec::with_capacity(NUM_MATTERS as usize);

    //FIXME use more readable code
    for i in 0..NUM_MATTERS {
        initial_matter_data.push(Matter {
            pos: Vec2::new((i % SIZE.0 as u32) as f32, (i / SIZE.0 as u32) as f32),
            vel: Vec2::new(0., 0.),
        });
    }

    let mut new_storage: GlobalStorage = GlobalStorage {
        buffers: HashMap::default(),
        stage_buffers: HashMap::default(),
    };
    new_storage.add_buffer("matter_src", &initial_matter_data, render_device.as_ref());
    new_storage.add_buffer("matter_dst", &initial_matter_data, render_device.as_ref());
    commands.insert_resource(new_storage);
}

pub struct PixelSimulationComputePlugin;

impl Plugin for PixelSimulationComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugin(ExtractResourcePlugin::<PixelSimulationImage>::default());
        app.add_plugin(ExtractResourcePlugin::<GlobalStorage>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(Render, queue_bind_group.in_set(RenderSet::Queue));

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("pixel_simulation", PixelSimulationNode::default());
        render_graph.add_node_edge(
            "pixel_simulation",
            bevy::render::main_graph::node::CAMERA_DRIVER,
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<PixelSimulationPipeline>();
    }
}

#[derive(Resource, Clone, Deref, ExtractResource)]
struct PixelSimulationImage(Handle<Image>);

#[derive(Resource)]
struct PixelSimulationBuffer(Buffer);

#[derive(Resource)]
struct PixelSimulationBindGroup(BindGroup);

fn copy_buffer(
    global_storage: Res<GlobalStorage>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    let matter_src = global_storage.buffers.get("matter_src").unwrap();
    let matter_dst = &global_storage
        .stage_buffers
        .get("matter_dst")
        .unwrap()
        .buffer;
    encoder.copy_buffer_to_buffer(matter_src, 0, matter_dst, 0, matter_dst.size());
    queue.submit(Some(encoder.finish()));
    device.wgpu_device().poll(MaintainBase::Wait);
}

fn unmap_all(mut global_storage: ResMut<GlobalStorage>) {
    global_storage.unmap_staging_buffers();
}

fn map_all(mut global_storage: ResMut<GlobalStorage>) {
    global_storage.map_staging_buffers();
}

fn on_click_compute(buttons: Res<Input<MouseButton>>, global_storage: Res<GlobalStorage>) {
    if buttons.just_pressed(MouseButton::Right) {
        let matter_dst = &global_storage.stage_buffers.get("matter_dst").unwrap();
        if matter_dst.mapped {
            let result =
                cast_slice::<u8, Matter>(&matter_dst.buffer.slice(..).get_mapped_range()).to_vec();
            result.iter().for_each(|m| {
                println!("{:?}", m);
            });
        }
    }
    if buttons.just_pressed(MouseButton::Left) {}
}

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<PixelSimulationPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    pixel_simulation_image: Res<PixelSimulationImage>,
    render_device: Res<RenderDevice>,
    storage: Res<GlobalStorage>,
) {
    let view = &gpu_images[&pixel_simulation_image.0];

    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.texture_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: storage
                    .buffers
                    .get("matter_src")
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: storage
                    .buffers
                    .get("matter_dst")
                    .unwrap()
                    .as_entire_binding(),
            },
        ],
    });
    commands.insert_resource(PixelSimulationBindGroup(bind_group));
}

#[derive(Resource)]
pub struct PixelSimulationPipeline {
    texture_bind_group_layout: BindGroupLayout,
    main_pipeline: CachedComputePipelineId,
}

impl FromWorld for PixelSimulationPipeline {
    fn from_world(world: &mut World) -> Self {
        let texture_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::Rgba8Unorm,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(Matter::min_size()),
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: Some(Matter::min_size()),
                            },
                            count: None,
                        },
                    ],
                });
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/pixel_simulation.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let main_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("main"),
        });

        PixelSimulationPipeline {
            texture_bind_group_layout,
            main_pipeline,
        }
    }
}

enum PixelSimulationState {
    Loading,
    Main,
}

struct PixelSimulationNode {
    state: PixelSimulationState,
}

impl Default for PixelSimulationNode {
    fn default() -> Self {
        Self {
            state: PixelSimulationState::Loading,
        }
    }
}

impl render_graph::Node for PixelSimulationNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<PixelSimulationPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            PixelSimulationState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.main_pipeline)
                {
                    self.state = PixelSimulationState::Main;
                }
            }
            PixelSimulationState::Main => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let texture_bind_group = &world.resource::<PixelSimulationBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<PixelSimulationPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, texture_bind_group, &[]);

        // select the pipeline based on the current state
        match self.state {
            PixelSimulationState::Loading => {}
            PixelSimulationState::Main => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.main_pipeline)
                    .unwrap();
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.0 as u32, SIZE.1 as u32, 1);
            }
        }

        Ok(())
    }
}

#[derive(Component)]
struct FpsText;

fn text_update_system(diagnostics: Res<DiagnosticsStore>, mut query: Query<&mut Text, With<FpsText>>) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                text.sections[1].value = format!("{value:.2}");
            }
        }
    }
}
