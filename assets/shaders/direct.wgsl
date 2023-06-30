const UP_LEFT = 0;
const UP = 1;
const UP_RIGHT = 2;
const RIGHT = 3;
const DOWN_RIGHT = 4;
const DOWN = 5;
const DOWN_LEFT = 6;
const LEFT = 7;

const OFFSETS:array<vec2<i32>, 8> = array<vec2<i32>, 8>(vec2<i32>(-1, 1), vec2<i32>(0, 1), vec2<i32>(1, 1), vec2<i32>(1, 0), vec2<i32>(1, -1), vec2<i32>(0, -1), vec2<i32>(-1, -1), vec2<i32>(-1, 0));