const canvas_size_x = 512.0;
const canvas_size_y = 512.0;
const empty_matter = 0x00000000;

struct Matter {
    pos: vec2<f32>,
    color: vec2<u32>,
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

fn get_top_matter_color(pos: vec2<f32>) -> vec2<u32> {
    let top_pos = vec2<f32>(pos.x, pos.y - 1.0);
    let new_index = top_pos.x + top_pos.y * canvas_size_x;
    if is_inside_canvas(top_pos) {
        return matter_src[u32(new_index)].color;
    } else {
        return vec2<u32>(u32(empty_matter), u32(0));
    }
}

fn get_down_matter_color(pos: vec2<f32>) -> vec2<u32> {
    let down_pos = vec2<f32>(pos.x, pos.y + 1.0);
    let new_index = down_pos.x + down_pos.y * canvas_size_x;
    if is_inside_canvas(down_pos) {
        return matter_src[u32(new_index)].color;
    } else {
        return vec2<u32>(u32(empty_matter), u32(0));
    }
}

fn is_empty(pos: vec2<f32>) -> bool {
    let index = u32(pos.x + pos.y * canvas_size_x);
    return matter_src[index].color.x == u32(empty_matter);
}

fn is_inside_canvas(pos: vec2<f32>) -> bool {
    return pos.x > 0.0 && pos.x < canvas_size_x && pos.y >= 0.0 && pos.y < canvas_size_y;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_matters = arrayLength(&matter_src);
    let index = global_id.x + global_id.y * u32(canvas_size_x);
    let location = vec2<i32>(i32(global_id.x), i32(global_id.y));
    var vPos = matter_src[index].pos;
    let top_pos = vec2<f32>(vPos.x, vPos.y - 1.0);
    let down_pos = vec2<f32>(vPos.x, vPos.y + 1.0);
    matter_dst[index].pos = vPos;

    if is_empty(vPos) {
        var vColor = get_top_matter_color(matter_src[index].pos);
        matter_dst[index].color = vColor;
    } else {
        if !is_empty(down_pos) {
            var vColor = matter_src[index].color;
            matter_dst[index].color = vColor;
        } else {
            if vPos.y == canvas_size_y - 1.0 {
                var vColor = matter_src[index].color;
                matter_dst[index].color = vColor;
            } else {
                var vColor = matter_src[index].color;
                matter_dst[index].color.x = u32(empty_matter);
            }
        }
    }



    storageBarrier();
    textureStore(texture, location, matter_color_to_vec4(matter_dst[index].color.x));
}