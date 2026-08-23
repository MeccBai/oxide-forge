use cuda_device::device;

#[device]
pub(super) fn random(mut x: u32) -> u32 {
    x += x << 13;
    x -= x >> 17;
    x *= x << 5;
    x
}
