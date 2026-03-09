module wgpu;

// Provide missing compiler-rt intrinsic for wasm32 (128-bit overflow multiply).
// The C3 compiler emits a reference to this symbol but does not link compiler-rt
// for the wasm32 target, so we supply a minimal implementation here.
fn int128 __muloti4(int128 a, int128 b, int* overflow) @export("__muloti4") {
    *overflow = 0;
    int128 result = a * b;
    if (a != 0 && b != 0) {
        if (a > 0 && b > 0 && result < 0) *overflow = 1;
        if (a < 0 && b < 0 && result < 0) *overflow = 1;
        if (a > 0 && b < 0 && result > 0) *overflow = 1;
        if (a < 0 && b > 0 && result > 0) *overflow = 1;
    }
    return result;
}
