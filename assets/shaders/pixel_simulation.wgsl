const canvas_size_x = 1024u;
const canvas_size_y = 1024u;
const empty_matter = 0x00000000u;
const DOWN_LEFT = 0;
const DOWN = 1;
const DOWN_RIGHT = 2;
const RIGHT = 3;
const UP_RIGHT = 4;
const UP = 5;
const UP_LEFT = 6;
const LEFT = 7;

const OFFSETS:array<vec2<i32>, 8> = array<vec2<i32>, 8>(vec2<i32>(-1, 1), vec2<i32>(0, 1), vec2<i32>(1, 1), vec2<i32>(1, 0), vec2<i32>(1, -1), vec2<i32>(0, -1), vec2<i32>(-1, -1), vec2<i32>(-1, 0));

struct Matter {
    color: u32,
}

@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

@group(0) @binding(1)
var<storage> matter_src : array<Matter>;

@group(0) @binding(2)
var<storage, read_write> matter_dst : array<Matter>;

@group(0) @binding(3)
var<storage, read_write> can_fall_map : array<u32>;

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

fn can_go_other_side(current_index: u32, to: vec2<u32>) {
}

fn calc_can_fall(col: u32, row: u32) {
    for (var i = row + 1; i < canvas_size_y - 1; i++) {
        let index = i32(i * canvas_size_x + col);
        if is_empty(index) {
            can_fall_map[col + row * canvas_size_x] = u32(1);
            break;
        }
    } 
}

fn can_go_down(index: i32) -> bool {
    return is_empty(index) || can_fall_map[index] != empty_matter;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = i32(global_id.x + global_id.y * canvas_size_x);
    let location = vec2<i32>(i32(global_id.x), i32(global_id.y));
    matter_dst[index].color = empty_matter;
    can_fall_map[index] = empty_matter;
    calc_can_fall(global_id.x, global_id.y);
    if can_go_down(index) {
        let up_index = get_neighbors_index(index, OFFSETS[UP]);
        let left_up_index = get_neighbors_index(index, OFFSETS[UP_LEFT]);
        let right_up_index = get_neighbors_index(index, OFFSETS[UP_RIGHT]);
        if !is_empty(up_index) {
            matter_dst[index].color = get_neighbors_color(index, OFFSETS[UP]) ;
        } else {
            if global_id.x != canvas_size_x - 1 && !is_empty(right_up_index) && !can_go_down(get_neighbors_index(right_up_index, OFFSETS[DOWN])) {
                matter_dst[index].color = get_neighbors_color(index, OFFSETS[UP_RIGHT]) ;
            } else {
                if global_id.x != 0 && !is_empty(left_up_index) && !can_go_down(get_neighbors_index(left_up_index, OFFSETS[DOWN])) && !can_go_down(get_neighbors_index(left_up_index, OFFSETS[DOWN_LEFT])) {
                    matter_dst[index].color = get_neighbors_color(index, OFFSETS[UP_LEFT]) ;
                }
            }
        }
    } else {
        let down_index = get_neighbors_index(index, OFFSETS[DOWN]);
        let left_down_index = get_neighbors_index(index, OFFSETS[DOWN_LEFT]);
        let right_down_index = get_neighbors_index(index, OFFSETS[DOWN_RIGHT]);
        let can_go_down = can_go_down(down_index);
        let can_go_left = global_id.x != 0 && can_go_down(left_down_index);
        let can_go_right = global_id.x != canvas_size_x -1 && can_go_down(right_down_index);
        if global_id.y != canvas_size_y - 1 && (can_go_down || can_go_left || can_go_right) {
            matter_dst[index].color = empty_matter;
        } else {
            matter_dst[index].color = matter_src[index].color;
        }
    }
    // storageBarrier();
    textureStore(texture, location, matter_color_to_vec4(matter_dst[index].color));
}