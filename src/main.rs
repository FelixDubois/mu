extern crate mu;

use mu::matrix::Mat;

fn main() {
    let mut m: Mat = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    println!("{:}", m);
}
