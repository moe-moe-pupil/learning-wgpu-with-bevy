const canvas_size_x = 512u;
const canvas_size_y = 512u;
const empty_matter = 0xffffffffu;
const empty_u32 = u32(0);
const DOWN_LEFT = 1;
const DOWN = 2;
const DOWN_RIGHT = 3;
const RIGHT = 4;
const UP_RIGHT = 5;
const UP = 6;
const UP_LEFT = 7;
const LEFT = 8;

const OFFSETS:array<vec2<i32>, 9> = array<vec2<i32>, 9>(vec2<i32>(0, 0), vec2<i32>(-1, 1), vec2<i32>(0, 1), vec2<i32>(1, 1), vec2<i32>(1, 0), vec2<i32>(1, -1), vec2<i32>(0, -1), vec2<i32>(-1, -1), vec2<i32>(-1, 0));

struct Matter {
    color: u32,
    lock: atomic<u32>
}

struct Line {
    prev_height: u32,
    height: u32,
    can_move: u32,
}

@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

@group(0) @binding(1)
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
        return matter_dst[neighbor_index].color;
    } else {
        return empty_matter;
    }
}

fn is_empty(index: i32) -> bool {
    return matter_dst[index].color == empty_matter;
}

fn is_inside_canvas(index: i32) -> bool {
    return index >= 0 && index < i32(canvas_size_x * canvas_size_y);
}

fn lock(index: i32) -> bool {
    let lock_ptr = &matter_dst[index].lock;
    let original_lock_value = atomicLoad(lock_ptr);
    if original_lock_value > 0u {
        return false;
    }
    return atomicAdd(lock_ptr, 1u) == original_lock_value;
}

fn is_not_empty_and_not_locked(index: i32) -> bool {
    return !is_empty(index) && !is_lock(index);
}

fn is_not_empty_and_locked(index: i32) -> bool {
    return !is_empty(index) && is_lock(index);
}

fn unlock(index: i32) {
    atomicStore(&matter_dst[index].lock, 0u);
}

fn is_lock(index: i32) -> bool {
    let lock_ptr = &matter_dst[index].lock;
    let original_lock_value = atomicLoad(lock_ptr);
    if original_lock_value > 0u {
        return true;
    }
    return false;
}

//TODO: pre-calc empty tile
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = i32(global_id.x + global_id.y * canvas_size_x);
    let location = vec2<i32>(i32(global_id.x), i32(global_id.y));
    unlock(index);
    storageBarrier();
    workgroupBarrier();
    if is_empty(index) && !is_lock(index) {
        let up_index = get_neighbors_index(index, OFFSETS[UP]);
        let left_up_index = get_neighbors_index(index, OFFSETS[UP_LEFT]);
        let right_up_index = get_neighbors_index(index, OFFSETS[UP_RIGHT]);
        if is_not_empty_and_not_locked(up_index) {
            var iter_index = index;
            while is_inside_canvas(iter_index) && is_empty(iter_index) {
                let up_index = get_neighbors_index(iter_index, OFFSETS[UP]);
                if is_inside_canvas(up_index) && is_not_empty_and_not_locked(up_index) {
                    lock(iter_index);
                    lock(up_index);
                    matter_dst[iter_index].color = matter_dst[up_index].color;
                    matter_dst[up_index].color = empty_matter;
                    iter_index = get_neighbors_index(iter_index, OFFSETS[UP]);
                } else {
                    break;
                }
            }
        } else {
            if global_id.x != canvas_size_x - 1 && is_not_empty_and_not_locked(right_up_index) && !is_empty(get_neighbors_index(right_up_index, OFFSETS[DOWN])) {
                lock(index);
                lock(right_up_index);
                matter_dst[index].color = get_neighbors_color(index, OFFSETS[UP_RIGHT]);
                matter_dst[right_up_index].color = empty_matter;
            }
            if global_id.x != 0 && is_not_empty_and_not_locked(left_up_index) && !is_empty(get_neighbors_index(left_up_index, OFFSETS[DOWN])){
                lock(index);
                lock(left_up_index);
                matter_dst[index].color = get_neighbors_color(index, OFFSETS[UP_LEFT]);
                matter_dst[left_up_index].color = empty_matter;
            }
        }
    }
    storageBarrier();
    textureStore(texture, location, matter_color_to_vec4(matter_dst[index].color));
}