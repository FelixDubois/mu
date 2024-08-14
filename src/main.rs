extern crate mu;

use mu::matrix::Mat;

fn main() {
    let m: Mat = Mat::from_vec(3, 3, vec![1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 5.0, 6.0, 1.0]);
    println!("{:}", m);

    println!("{:}", m.det());
}
