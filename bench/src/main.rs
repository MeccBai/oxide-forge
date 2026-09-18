use oxide_forge::cuda::{
    CudaRuntime,
    InitType::{Random, Reserve},
};
use std::time::Instant;

const SIZE: usize = 512;

fn main() {
    let mut runtime = CudaRuntime::new().unwrap();
    let mat1 = runtime.new_matrix(Random, SIZE, SIZE, None);

    let matc = runtime.new_matrix(Random, SIZE, SIZE, None);

    let splits: [usize; 2] = [256, 256];

    let add = move |x: f32, y: f32| -> f32 { x + y };

    let mat1s = runtime.matrix_split_rows(&mat1, &splits, None);

    println!("mat1s[0] shape: {:?}", mat1s[0].shape());

    println!("matc shape: {:?}", matc.shape());

    let streams = runtime.create_extra_streams(2);

    let mat1_c_s = [
        runtime.matrix_multiply(&mat1s[0], &matc, Some(&streams[0])),
        runtime.matrix_multiply(&mat1s[1], &matc, Some(&streams[1])),
    ];
    runtime.join_streams(&streams);

    let mut mat1_c_r = [&mat1_c_s[0], &mat1_c_s[1]];

    let mut mat1_c = runtime.matrix_concat_rows(&mat1_c_r, None);

    runtime.sync();

    println!("mat1_c shape: {:?}", mat1_c.shape());

    let mat2 = runtime.new_matrix(Random, SIZE, SIZE, None);

    let mat2_s = runtime.matrix_split_cols(&mat2, &splits, None);
    let matc_s = runtime.matrix_split_rows(&matc, &splits, None);

    println!("mat2_s[0] shape: {:?}", mat2_s[0].shape());
    println!("matc_s[0] shape: {:?}", matc_s[0].shape());

    let mut mat2_c_s1 = runtime.matrix_multiply(&mat2_s[0], &matc_s[0], None);
    let mut mat2_c_s2 = runtime.matrix_multiply(&mat2_s[1], &matc_s[1], None);

    mat2_c_s1.binary_assign(&mat2_c_s2, add, &runtime);

    println!("mat2_c_s[0] shape: {:?}", mat2_c_s1.shape());
}
