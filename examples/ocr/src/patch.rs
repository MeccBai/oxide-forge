use image::{GrayImage, RgbImage};

pub const IMAGE_SIZE: usize = 512;
pub const PATCH_SIZE: usize = 16;
pub const PATCHES_PER_SIDE: usize = IMAGE_SIZE / PATCH_SIZE;
pub const PATCH_COUNT: usize = PATCHES_PER_SIDE * PATCHES_PER_SIDE;
pub const IMAGE_CHANNELS: usize = 3;
pub const INPUT_WIDTH: usize = PATCH_SIZE * PATCH_SIZE * IMAGE_CHANNELS;
pub const TARGET_WIDTH: usize = PATCH_SIZE * PATCH_SIZE;

pub fn encode_image(image: &RgbImage) -> Vec<f32> {
    assert_eq!(image.width() as usize, IMAGE_SIZE);
    assert_eq!(image.height() as usize, IMAGE_SIZE);

    let mut patches = vec![0.0; PATCH_COUNT * INPUT_WIDTH];
    for patch_y in 0..PATCHES_PER_SIDE {
        for patch_x in 0..PATCHES_PER_SIDE {
            let patch = patch_y * PATCHES_PER_SIDE + patch_x;
            let row = &mut patches[patch * INPUT_WIDTH..(patch + 1) * INPUT_WIDTH];
            for local_y in 0..PATCH_SIZE {
                for local_x in 0..PATCH_SIZE {
                    let pixel = image.get_pixel(
                        (patch_x * PATCH_SIZE + local_x) as u32,
                        (patch_y * PATCH_SIZE + local_y) as u32,
                    );
                    let pixel_offset = (local_y * PATCH_SIZE + local_x) * IMAGE_CHANNELS;
                    for channel in 0..IMAGE_CHANNELS {
                        row[pixel_offset + channel] = pixel[channel] as f32 / 127.5 - 1.0;
                    }
                }
            }
        }
    }
    patches
}

pub fn encode_mask(mask: &GrayImage) -> Vec<f32> {
    assert_eq!(mask.width() as usize, IMAGE_SIZE);
    assert_eq!(mask.height() as usize, IMAGE_SIZE);

    let mut patches = vec![0.0; PATCH_COUNT * TARGET_WIDTH];
    for patch_y in 0..PATCHES_PER_SIDE {
        for patch_x in 0..PATCHES_PER_SIDE {
            let patch = patch_y * PATCHES_PER_SIDE + patch_x;
            let row = &mut patches[patch * TARGET_WIDTH..(patch + 1) * TARGET_WIDTH];
            for local_y in 0..PATCH_SIZE {
                for local_x in 0..PATCH_SIZE {
                    let pixel = mask.get_pixel(
                        (patch_x * PATCH_SIZE + local_x) as u32,
                        (patch_y * PATCH_SIZE + local_y) as u32,
                    );
                    row[local_y * PATCH_SIZE + local_x] = pixel[0] as f32 / 255.0;
                }
            }
        }
    }
    patches
}

pub fn decode_mask(patches: &[f32]) -> GrayImage {
    assert_eq!(patches.len(), PATCH_COUNT * TARGET_WIDTH);

    let mut mask = GrayImage::new(IMAGE_SIZE as u32, IMAGE_SIZE as u32);
    for patch_y in 0..PATCHES_PER_SIDE {
        for patch_x in 0..PATCHES_PER_SIDE {
            let patch = patch_y * PATCHES_PER_SIDE + patch_x;
            let row = &patches[patch * TARGET_WIDTH..(patch + 1) * TARGET_WIDTH];
            for local_y in 0..PATCH_SIZE {
                for local_x in 0..PATCH_SIZE {
                    let value = row[local_y * PATCH_SIZE + local_x].clamp(0.0, 1.0);
                    mask.put_pixel(
                        (patch_x * PATCH_SIZE + local_x) as u32,
                        (patch_y * PATCH_SIZE + local_y) as u32,
                        image::Luma([(value * 255.0).round() as u8]),
                    );
                }
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_patch_order_round_trips() {
        let mask = GrayImage::from_fn(IMAGE_SIZE as u32, IMAGE_SIZE as u32, |x, y| {
            image::Luma([((x + y * 3) % 256) as u8])
        });
        let decoded = decode_mask(&encode_mask(&mask));
        assert_eq!(mask, decoded);
    }
}
