use bevy::{
    core::Pod,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::{RenderAssetUsages, RenderAssets},
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            encase::{private::WriteInto, StorageBuffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Render, RenderApp, RenderSet,
    },
    utils::HashMap,
    window::{close_on_esc, PrimaryWindow, WindowMode, WindowPlugin},
};
use bytemuck::{cast_slice, Zeroable};
use line_drawing;
use std::{borrow::Cow, num::NonZeroU64, println};
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct PixelSimulationLabel;
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

#[derive(Resource)]
pub struct RenderContextStorage {
    encoder: Option<CommandEncoder>,
    queue: RenderQueue,
    device: RenderDevice,
    submission_queue_processed: bool,
}

#[derive(Clone, Debug)]
pub struct StagingBuffer {
    mapped: bool,
    buffer: Buffer,
}

impl RenderContextStorage {
    fn poll(&mut self) -> bool {
        match self.device.wgpu_device().poll(wgpu::MaintainBase::Poll) {
            // The first few times the poll occurs the queue will be empty, because wgpu hasn't started anything yet.
            // We need to wait until `MaintainResult::Ok`, which means wgpu has started to process our data.
            // Then, the next time the queue is empty (`MaintainResult::SubmissionQueueEmpty`), wgpu has finished processing the data and we are done.
            wgpu::MaintainResult::SubmissionQueueEmpty => {
                let res = self.submission_queue_processed;
                self.submission_queue_processed = false;
                res
            }
            wgpu::MaintainResult::Ok => {
                self.submission_queue_processed = true;
                false
            }
        }
    }

    fn submit(&mut self) -> &mut Self {
        let encoder = self.encoder.take().unwrap();
        self.queue.submit(Some(encoder.finish()));
        self
    }

    fn copy_buffer(&mut self, src: &Buffer, dst: &Buffer, size: u64) -> &mut Self {
        match self.encoder.as_mut() {
            None => {
                self.encoder = Some(
                    self.device
                        .create_command_encoder(&CommandEncoderDescriptor { label: None }),
                );
                self.encoder
                    .as_mut()
                    .unwrap()
                    .copy_buffer_to_buffer(src, 0, dst, 0, size);
            }
            Some(encoder) => {
                encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
            }
        }
        self
    }
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

            read_buffer_slice.map_async(wgpu::MapMode::Read, |result| {
                if let Some(err) = result.err() {
                    panic!("{}", err.to_string());
                }
            });
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

    #[inline]
    fn swap(&mut self) {
        let [buffer_a, buffer_b] = self
            .buffers
            .get_many_mut(["matter_src", "matter_dst"])
            .unwrap();
        std::mem::swap(buffer_a, buffer_b);
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
    color: u32,
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

fn update_state(state: ResMut<State<BufferState>>, mut next_state: ResMut<NextState<BufferState>>) {
    match state.get() {
        BufferState::FinishedWorking => next_state.set(BufferState::Available),
        _ => {}
    }
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
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                mode: WindowMode::Windowed,
                ..default()
            }),
            ..default()
        }))
        .init_state::<BufferState>()
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            PixelSimulationComputePlugin,
        ))
        .add_systems(Update, close_on_esc)
        .add_systems(Update, text_update_system)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                update_state,
                (unmap_all, swap, copy_buffer, submit, map_all)
                    .chain()
                    .run_if(not(in_state::<BufferState>(BufferState::Working))),
                is_poll,
                on_click_compute,
            )
                .chain(),
        )
        .run();
}

fn is_poll(
    mut global_storage: ResMut<GlobalStorage>,
    mut render_storage: ResMut<RenderContextStorage>,
    state: ResMut<State<BufferState>>,
    mut next_state: ResMut<NextState<BufferState>>,
) {
    if render_storage.poll() {
        for (_, staging_buffer) in global_storage.stage_buffers.iter_mut() {
            // By this the staging buffers would've been mapped.
            staging_buffer.mapped = true;
        }
        next_state.set(BufferState::FinishedWorking)
    }
}
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
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
        RenderAssetUsages::default(),
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

    commands.insert_resource(PixelSimulationImage { texture: image });

    let font = asset_server.load::<Font>("fonts/NotoSansTC-Medium.otf");
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
                    color: Color::BEIGE,
                },
            ),
        ])
        .with_text_justify(JustifyText::Left)
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
            color: 0x00000000u32,
        });
    }

    let mut new_storage: GlobalStorage = GlobalStorage {
        buffers: HashMap::default(),
        stage_buffers: HashMap::default(),
    };
    new_storage.add_buffer("matter_src", &initial_matter_data, render_device.as_ref());
    new_storage.add_buffer("matter_dst", &initial_matter_data, render_device.as_ref());
    commands.insert_resource(RenderContextStorage {
        encoder: Some(
            render_device.create_command_encoder(&CommandEncoderDescriptor { label: None }),
        ),
        queue: render_queue.clone(),
        device: render_device.clone(),
        submission_queue_processed: false,
    });
    commands.insert_resource(new_storage);
}

pub struct PixelSimulationComputePlugin;

fn is_point_in_canvas(point: Vec2) -> bool {
    point.x >= 0.0 && point.y >= 0.0 && point.x < SIZE.0 && point.y < SIZE.1
}

impl Plugin for PixelSimulationComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins((
            ExtractResourcePlugin::<PixelSimulationImage>::default(),
            ExtractResourcePlugin::<GlobalStorage>::default(),
        ));
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            queue_bind_group.in_set(RenderSet::PrepareBindGroups),
        );

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node(PixelSimulationLabel, PixelSimulationNode::default());
        render_graph.add_node_edge(PixelSimulationLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<PixelSimulationPipeline>();
    }
}

#[derive(Resource, Clone, Deref, ExtractResource, AsBindGroup)]
struct PixelSimulationImage {
    #[storage_texture(0, image_format = Rgba8Unorm, access = ReadWrite)]
    texture: Handle<Image>,
}

#[derive(Resource)]
struct PixelSimulationBindGroup(BindGroup);

fn copy_buffer(
    mut global_storage: ResMut<GlobalStorage>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut render_storage: ResMut<RenderContextStorage>,
) {
    let matter_src = global_storage.buffers.get("matter_dst").unwrap();
    let matter_dst = &global_storage
        .stage_buffers
        .get("matter_dst")
        .unwrap()
        .buffer;
    render_storage.copy_buffer(matter_src, matter_dst, matter_dst.size());

    let matter_src = global_storage.buffers.get("matter_src").unwrap();
    let matter_dst = &global_storage
        .stage_buffers
        .get("matter_src")
        .unwrap()
        .buffer;
    render_storage.copy_buffer(matter_src, matter_dst, matter_dst.size());
}

fn submit(
    mut next_state: ResMut<NextState<BufferState>>,
    mut render_storage: ResMut<RenderContextStorage>,
) {
    render_storage.submit();
    next_state.set(BufferState::Working)
}

fn swap(
    mut global_storage: ResMut<GlobalStorage>,
    device: Res<RenderDevice>,
    mut render_storage: ResMut<RenderContextStorage>,
) {
    global_storage.swap();
}

fn unmap_all(mut global_storage: ResMut<GlobalStorage>) {
    global_storage.unmap_staging_buffers();
}

fn map_all(mut global_storage: ResMut<GlobalStorage>) {
    global_storage.map_staging_buffers();
}

fn on_click_compute(
    buttons: Res<ButtonInput<MouseButton>>,
    global_storage: Res<GlobalStorage>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
    queue: Res<RenderQueue>,
) {
    if let Some(position) = q_windows.single().cursor_position() {
        if buttons.just_pressed(MouseButton::Right) {
            let matter_dst = global_storage.stage_buffers.get("matter_dst").unwrap();
            if matter_dst.mapped {
                let result =
                    cast_slice::<u8, Matter>(&matter_dst.buffer.slice(..).get_mapped_range())
                        .to_vec();
                result.iter().for_each(|m| {
                    println!("{:?}", m);
                });
            }
        }
        if buttons.pressed(MouseButton::Left) {
            let matter_dst = global_storage.stage_buffers.get("matter_src").unwrap();
            let radius = 5;
            if matter_dst.mapped {
                let mut result =
                    cast_slice::<u8, Matter>(&matter_dst.buffer.slice(..).get_mapped_range())
                        .to_vec();
                for x in position.as_ivec2().x - radius..=position.as_ivec2().x + radius {
                    for y in position.as_ivec2().y - radius..=position.as_ivec2().y + radius {
                        if is_point_in_canvas(Vec2 {
                            x: x as f32,
                            y: y as f32,
                        }) {
                            let index = (x + y * SIZE.0 as i32) as usize;
                            result[index] = Matter {
                                color: 0xffffffffu32,
                            };
                        }
                    }
                }

                queue.write_buffer(
                    global_storage.buffers.get("matter_src").unwrap(),
                    0,
                    cast_slice(&result),
                );
            }
        }
    }
}

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<PixelSimulationPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    pixel_simulation_image: Res<PixelSimulationImage>,
    render_device: Res<RenderDevice>,
    storage: Res<GlobalStorage>,
) {
    let view = &gpu_images.get(&pixel_simulation_image.texture).unwrap();

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &[
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
    );
    commands.insert_resource(PixelSimulationBindGroup(bind_group));
}

#[derive(Resource)]
pub struct PixelSimulationPipeline {
    texture_bind_group_layout: BindGroupLayout,
    main_pipeline: CachedComputePipelineId,
}

impl FromWorld for PixelSimulationPipeline {
    fn from_world(world: &mut World) -> Self {
        let texture_bind_group_layout = world.resource::<RenderDevice>().create_bind_group_layout(
            None,
            &[
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
        );
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

#[derive(States, Debug, Default, Clone, Eq, PartialEq, Hash)]
pub enum BufferState {
    #[default]
    Created,
    Available,
    Working,
    FinishedWorking,
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
                pass.dispatch_workgroups(SIZE.0 as u32 / 16, SIZE.1 as u32 / 16, 1);
            }
        }

        Ok(())
    }
}

#[derive(Component)]
struct FpsText;

fn text_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                text.sections[1].value = format!("{value:.2}");
            }
        }
    }
}
