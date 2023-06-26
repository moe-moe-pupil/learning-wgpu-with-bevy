const canvas_size_x = 1;
const canvas_size_y = 1;
const empty_matter = 1;

struct Matter {
    pos: vec2<f32>,
    color: vec2<f32>,
}

@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

@group(0) @binding(1)
var<storage> matter_src : array<Matter>;

@group(0) @binding(2)
var<storage, read_write> matter_dst : array<Matter>;

fn matter_color_to_vec4(color: u32) -> vec4<f32> {
    var r = f32((color >> u32(24)) & u32(255)) / 255.0;
    var g = f32((color >> u32(16)) & u32(255)) / 255.0;
    var b = f32((color >> u32(8)) & u32(255)) / 255.0;
    return vec4<f32>(r, g, b, 1.0);
}

fn linear_from_srgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(10.31475);
    let a: vec3<f32> = select(vec3<f32>(0.0), vec3<f32>(1.0), cutoff);
    let lower = srgb / vec3(3294.6);
    let higher = pow((srgb + vec3<f32>(14.025)) / vec3<f32>(269.026), vec3<f32>(2.4));
    return mix(lower, higher, a);
}

fn linear_from_srgba(srgba: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(linear_from_srgb(srgba.rgb * 255.0), srgba.a);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_matters = arrayLength(&matter_src);
    let index = global_id.x + global_id.y * u32(512);

    var vPos = matter_src[index].pos;
    var vColor = matter_src[index].color;
    matter_dst[index].pos = vPos;
    matter_dst[index].color = vColor;

    storageBarrier();
    let location = vec2<i32>(i32(global_id.x), i32(global_id.y));
    textureStore(texture, location, matter_color_to_vec4(u32(matter_dst[index].color.x)));
}