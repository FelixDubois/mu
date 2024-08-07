mod matrix {
    use std::f64;
    use std::fmt;
    use std::ops;

    #[derive(Clone, PartialEq)]
    pub struct Mat {
        data: Vec<f64>,
        rows: usize,
        cols: usize,
    }

    pub fn new(rows: usize, cols: usize) -> Mat {
        assert!(rows != 0 && cols != 0, "Cant create an empty matrix!");

        let data = vec![0.0; rows * cols];
        return Mat { data, rows, cols };
    }

    impl Mat {
        pub fn new(rows: usize, cols: usize) -> Mat {
            assert!(rows != 0 && cols != 0, "Cant create an empty matrix!");

            let data = vec![0.0; rows * cols];
            return Mat { data, rows, cols };
        }

        pub fn new_from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Mat {
            assert_eq!(
                rows * cols,
                data.len(),
                "Data length must match matrix size."
            );
            return Mat { data, rows, cols };
        }

        pub fn new_filled(rows: usize, cols: usize, value: f64) -> Mat {
            let data = vec![value; rows * cols];
            return Mat { data, rows, cols };
        }

        pub fn zeros(rows: usize, cols: usize) -> Mat {
            return new(rows, cols);
        }

        pub fn square(size: usize) -> Mat {
            return Mat::zeros(size, size);
        }

        pub fn ones(rows: usize, cols: usize) -> Mat {
            return Mat::new_filled(rows, cols, 1.0);
        }

        pub fn eye(size: usize) -> Mat {
            let mut m = Mat::square(size);
            for i in 0..size {
                m[(i, i)] = 1.0;
            }
            return m;
        }

        pub fn shape(&self) -> (usize, usize) {
            return (self.rows, self.cols);
        }

        pub fn transpose(&self) -> Mat {
            let mut m = Mat::zeros(self.cols, self.rows);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    m[(j, i)] = self[(i, j)];
                }
            }
            return m;
        }

        pub fn trace(&self) -> f64 {
            assert_eq!(self.rows, self.cols, "Matrix must be square.");

            let mut sum = 0.0;
            for i in 0..self.rows {
                sum += self[(i, i)];
            }

            return sum;
        }

        pub fn dot(&self, b: Mat) -> Mat {
            assert_eq!(self.cols, b.rows, "Matrix dimensions not compatible.");

            return self.clone() * b;
        }

        pub fn sub_matrix(&self, row: usize, col: usize) -> Mat {
            assert!(row < self.rows, "Row index out of bounds.");
            assert!(col < self.cols, "Column index out of bounds.");
            assert!(
                self.cols > 1 && self.rows > 1,
                "Matrix must be at least 2x2."
            );

            let mut m = Mat::zeros(self.rows - 1, self.cols - 1);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    if i < row && j < col {
                        m[(i, j)] = self[(i, j)];
                    } else if i < row && j > col {
                        m[(i, j - 1)] = self[(i, j)];
                    } else if i > row && j < col {
                        m[(i - 1, j)] = self[(i, j)];
                    } else if i > row && j > col {
                        m[(i - 1, j - 1)] = self[(i, j)];
                    }
                }
            }
            return m;
        }

        // https://en.wikipedia.org/wiki/Determinant
        pub fn det(&self) -> f64 {
            assert_eq!(self.rows, self.cols, "Matrix must be squared!");

            if self.rows == 2 {
                return self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)];
            }

            let mut det = 0.0;
            for i in 0..self.cols {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                det += sign * self[(0, i)] * self.sub_matrix(0, i).det();
            }

            return det;
        }

        pub fn pow(&self, n: i32) -> Mat {
            assert_eq!(self.rows, self.cols, "Matrix must be squared!");
            assert!(n >= 0, "Exponent must be non-negative.");

            if n == 0 {
                return Mat::eye(self.rows);
            }

            if n == 1 {
                return self.clone();
            }

            let mut m = self.clone();
            for _ in 1..n {
                m = m.clone() * self.clone();
            }

            return m;
        }

        // https://en.wikipedia.org/wiki/Cofactor_matrix
        pub fn cofactor(&self) -> Mat {
            assert_eq!(self.rows, self.cols, "Matrix must be squared!");

            let mut adj = Mat::zeros(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let sub_mat = self.sub_matrix(i, j);
                    let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };

                    adj[(i, j)] = sign * sub_mat.det();
                }
            }

            return adj;
        }

        // https://en.wikipedia.org/wiki/Adjugate_matrix
        pub fn adjugate(&self) -> Mat {
            assert_eq!(self.rows, self.cols, "Matrix must be squared!");

            return self.cofactor().transpose();
        }

        // https://en.wikipedia.org/wiki/Invertible_matrix
        pub fn inverse(&self) -> Mat {
            let det: f64 = self.det();

            assert!(det != 0.0, "The determinant must be different than 0");
            let cofactor = self.adjugate();
            return cofactor / det;
        }
    }

    impl fmt::Display for Mat {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    write!(f, "{:.2} ", self[(i, j)])?;
                }
                write!(f, "\n")?;
            }
            return Ok(());
        }
    }

    impl fmt::Debug for Mat {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Mat {{ rows: {}, cols: {} }}\n", self.rows, self.cols)?;
            for i in 0..self.rows {
                for j in 0..self.cols {
                    write!(f, "{:.2} ", self[(i, j)])?;
                }
                write!(f, "\n")?;
            }
            return Ok(());
        }
    }

    impl ops::Index<(usize, usize)> for Mat {
        type Output = f64;

        fn index(&self, index: (usize, usize)) -> &f64 {
            let (i, j) = index;
            return &self.data[i * self.cols + j];
        }
    }

    impl ops::IndexMut<(usize, usize)> for Mat {
        fn index_mut(&mut self, index: (usize, usize)) -> &mut f64 {
            let (i, j) = index;
            return &mut self.data[i * self.cols + j];
        }
    }

    impl ops::Add for Mat {
        type Output = Mat;

        fn add(self, other: Mat) -> Mat {
            assert_eq!(
                self.rows, other.rows,
                "Matrix must have the same number of rows."
            );
            assert_eq!(
                self.cols, other.cols,
                "Matrix must have the same number of columns."
            );

            let mut m = Mat::zeros(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    m[(i, j)] = self[(i, j)] + other[(i, j)];
                }
            }
            return m;
        }
    }

    impl ops::AddAssign for Mat {
        fn add_assign(&mut self, other: Mat) {
            assert_eq!(self.rows, other.rows);
            assert_eq!(self.cols, other.cols);

            for i in 0..self.rows {
                for j in 0..self.cols {
                    self[(i, j)] += other[(i, j)];
                }
            }
        }
    }

    impl ops::Sub for Mat {
        type Output = Mat;

        fn sub(self, other: Mat) -> Mat {
            assert_eq!(self.rows, other.rows);
            assert_eq!(self.cols, other.cols);

            let mut m = Mat::zeros(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    m[(i, j)] = self[(i, j)] - other[(i, j)];
                }
            }
            return m;
        }
    }

    impl ops::SubAssign for Mat {
        fn sub_assign(&mut self, other: Mat) {
            assert_eq!(self.rows, other.rows);
            assert_eq!(self.cols, other.cols);

            for i in 0..self.rows {
                for j in 0..self.cols {
                    self[(i, j)] -= other[(i, j)];
                }
            }
        }
    }

    impl ops::Mul for Mat {
        type Output = Mat;

        fn mul(self, other: Mat) -> Mat {
            assert_eq!(self.cols, other.rows);

            let mut m = Mat::zeros(self.rows, other.cols);
            for i in 0..self.rows {
                for j in 0..other.cols {
                    for k in 0..self.cols {
                        m[(i, j)] += self[(i, k)] * other[(k, j)];
                    }
                }
            }

            return m;
        }
    }

    impl ops::Mul<Mat> for f64 {
        type Output = Mat;

        fn mul(self, rhs: Mat) -> Mat {
            let mut m = rhs.clone();

            for i in 0..rhs.rows {
                for j in 0..rhs.cols {
                    m[(i, j)] *= self;
                }
            }

            return m;
        }
    }

    impl ops::Mul<Mat> for u32 {
        type Output = Mat;

        fn mul(self, rhs: Mat) -> Self::Output {
            return f64::from(self) * rhs;
        }
    }

    impl ops::Div<f64> for Mat {
        type Output = Mat;

        fn div(self, rhs: f64) -> Mat {
            assert!(rhs != 0.0, "Can`'t divide by 0!");
            let mut m = self.clone();

            for i in 0..self.rows {
                for j in 0..self.cols {
                    m[(i, j)] /= rhs;
                }
            }

            return m;
        }
    }

    impl ops::MulAssign for Mat {
        fn mul_assign(&mut self, other: Mat) {
            assert_eq!(self.cols, other.rows);

            let mut m = Mat::zeros(self.rows, other.cols);
            for i in 0..self.rows {
                for j in 0..other.cols {
                    for k in 0..self.cols {
                        m[(i, j)] += self[(i, k)] * other[(k, j)];
                    }
                }
            }

            *self = m;
        }
    }

    impl ops::Neg for Mat {
        type Output = Mat;

        fn neg(self) -> Mat {
            let mut m = Mat::zeros(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    m[(i, j)] = -self[(i, j)];
                }
            }
            return m;
        }
    }
}
