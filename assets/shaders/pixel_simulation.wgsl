const canvas_size_x = 512u;
const canvas_size_y = 512u;
const empty_matter = 0x00000000u;

struct Matter {
    color: u32,
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

fn get_neighbors_index(index: i32, to: vec2<i32>) -> i32 {
    return index + to.x + to.y * i32(canvas_size_x);
}

fn get_neighbors_color(index: i32, to: vec2<i32>) -> u32 {
    let neighbor_index = get_neighbors_index(index, to);
    if is_inside_canvas(neighbor_index) {
        return matter_src[neighbor_index].color;
    } else {
        return empty_matter;
    }
}

fn is_empty(index: i32) -> bool {
    return matter_src[index].color == empty_matter;
}

fn is_inside_canvas(index: i32) -> bool {
    return index >= 0 && index < i32(canvas_size_x * canvas_size_y);
}

fn get_neighbors(current_index: u32, to: vec2<u32>) {
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = i32(global_id.x + global_id.y * canvas_size_x);
    let location = vec2<i32>(i32(global_id.x), i32(global_id.y));

    if is_empty(index) {
        matter_dst[index].color = get_neighbors_color(index, vec2<i32>(0, -1)) ;
    } else {
        if !is_empty(get_neighbors_index(index, vec2<i32>(0, 1))) || global_id.y == canvas_size_y - 1u {
            matter_dst[index].color = matter_src[index].color;
        } else {
            matter_dst[index].color = empty_matter;
        }
    }
    storageBarrier();
    textureStore(texture, location, matter_color_to_vec4(matter_dst[index].color));
}