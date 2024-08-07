use std::f64;
use std::fmt;
use std::ops;

#[derive(Clone, PartialEq, Debug)]
pub struct Mat {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows != 0 && cols != 0, "Can't create an empty matrix!");
        let data = vec![0.0; rows * cols];
        Self { data, rows, cols }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            rows * cols,
            data.len(),
            "Data length must match matrix size."
        );
        Self { data, rows, cols }
    }

    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        let data = vec![value; rows * cols];
        Self { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::filled(rows, cols, 1.0)
    }

    pub fn eye(size: usize) -> Self {
        let mut m = Self::zeros(size, size);
        for i in 0..size {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn transpose(&self) -> Self {
        let mut m = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                m[(j, i)] = self[(i, j)];
            }
        }
        m
    }

    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols, "Matrix must be square.");
        (0..self.rows).map(|i| self[(i, i)]).sum()
    }

    pub fn dot(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "Matrix dimensions not compatible.");
        self * other
    }

    pub fn sub_matrix(&self, row: usize, col: usize) -> Self {
        assert!(row < self.rows, "Row index out of bounds.");
        assert!(col < self.cols, "Column index out of bounds.");
        assert!(
            self.cols > 1 && self.rows > 1,
            "Matrix must be at least 2x2."
        );

        let mut m = Self::zeros(self.rows - 1, self.cols - 1);
        for (i, r) in (0..self.rows).filter(|&r| r != row).enumerate() {
            for (j, c) in (0..self.cols).filter(|&c| c != col).enumerate() {
                m[(i, j)] = self[(r, c)];
            }
        }
        m
    }

    pub fn det(&self) -> f64 {
        assert_eq!(self.rows, self.cols, "Matrix must be square!");

        match self.rows {
            1 => self[(0, 0)],
            2 => self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)],
            _ => (0..self.cols)
                .map(|i| {
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    sign * self[(0, i)] * self.sub_matrix(0, i).det()
                })
                .sum(),
        }
    }

    pub fn pow(&self, n: u32) -> Self {
        assert_eq!(self.rows, self.cols, "Matrix must be square!");

        match n {
            0 => Self::eye(self.rows),
            1 => self.clone(),
            _ => {
                let half = self.pow(n / 2);
                let result = &half * &half;
                if n % 2 == 0 {
                    result
                } else {
                    &result * self
                }
            }
        }
    }

    pub fn cofactor(&self) -> Self {
        assert_eq!(self.rows, self.cols, "Matrix must be square!");

        let mut adj = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let sub_mat = self.sub_matrix(i, j);
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                adj[(i, j)] = sign * sub_mat.det();
            }
        }
        adj
    }

    pub fn adjugate(&self) -> Self {
        self.cofactor().transpose()
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.det();
        if det == 0.0 {
            None
        } else {
            Some(&self.adjugate() / det)
        }
    }
}

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:.2} ", self[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl ops::Index<(usize, usize)> for Mat {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.data[i * self.cols + j]
    }
}

impl ops::IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.data[i * self.cols + j]
    }
}

impl ops::Add for &Mat {
    type Output = Mat;

    fn add(self, other: &Mat) -> Mat {
        assert_eq!(
            self.rows, other.rows,
            "Matrices must have the same dimensions"
        );
        assert_eq!(
            self.cols, other.cols,
            "Matrices must have the same dimensions"
        );

        let mut result = Mat::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = self[(i, j)] + other[(i, j)];
            }
        }
        result
    }
}

impl ops::Sub for &Mat {
    type Output = Mat;

    fn sub(self, other: &Mat) -> Mat {
        assert_eq!(
            self.rows, other.rows,
            "Matrices must have the same dimensions"
        );
        assert_eq!(
            self.cols, other.cols,
            "Matrices must have the same dimensions"
        );

        let mut result = Mat::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = self[(i, j)] - other[(i, j)];
            }
        }
        result
    }
}

impl ops::Mul for &Mat {
    type Output = Mat;

    fn mul(self, other: &Mat) -> Mat {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions not compatible for multiplication"
        );

        let mut result = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                result[(i, j)] = (0..self.cols).map(|k| self[(i, k)] * other[(k, j)]).sum();
            }
        }
        result
    }
}

impl ops::Mul<f64> for &Mat {
    type Output = Mat;

    fn mul(self, scalar: f64) -> Mat {
        let mut result = self.clone();
        for val in result.data.iter_mut() {
            *val *= scalar;
        }
        result
    }
}

impl ops::Mul<&Mat> for f64 {
    type Output = Mat;

    fn mul(self, m: &Mat) -> Mat {
        let mut result = m.clone();
        for val in result.data.iter_mut() {
            *val *= self;
        }
        result
    }
}

impl ops::Div<f64> for &Mat {
    type Output = Mat;

    fn div(self, scalar: f64) -> Mat {
        assert!(scalar != 0.0, "Cannot divide by zero");
        let mut result = self.clone();
        for val in result.data.iter_mut() {
            *val /= scalar;
        }
        result
    }
}

impl ops::Neg for &Mat {
    type Output = Mat;

    fn neg(self) -> Mat {
        let mut result = self.clone();
        for val in result.data.iter_mut() {
            *val = -(*val);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

    #[test]
    fn test_new() {
        let m = Mat::new(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data, vec![0.0; 6]);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let m = Mat::from_vec(2, 2, data.clone());
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.data, data);
    }

    #[test]
    fn test_zeros() {
        let m = Mat::zeros(3, 2);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 2);
        assert!(m.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let m = Mat::ones(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert!(m.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_eye() {
        let m = Mat::eye(3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(m[(i, j)], 1.0);
                } else {
                    assert_eq!(m[(i, j)], 0.0);
                }
            }
        }
    }
    #[test]
    fn test_transpose() {
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mt = m.transpose();
        assert_eq!(mt.rows, 3);
        assert_eq!(mt.cols, 2);
        assert_eq!(mt.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_trace() {
        let m = Mat::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(m.trace(), 15.0);
    }

    #[test]
    fn test_dot() {
        let m1 = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m2 = Mat::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = m1.dot(&m2);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_det() {
        let m = Mat::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!((m.det() - 0.0).abs() < EPSILON);

        let m2 = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m2.det(), -2.0);
    }

    #[test]
    fn test_inverse() {
        let m = Mat::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse().unwrap();
        let expected = Mat::from_vec(2, 2, vec![0.6, -0.7, -0.2, 0.4]);
        for i in 0..2 {
            for j in 0..2 {
                assert!((inv[(i, j)] - expected[(i, j)]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_add() {
        let m1 = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Mat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = &m1 + &m2;
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_sub() {
        let m1 = Mat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let m2 = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = &m1 - &m2;
        assert_eq!(result.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_mul() {
        let m1 = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Mat::from_vec(2, 2, vec![2.0, 0.0, 1.0, 2.0]);
        let result = &m1 * &m2;
        assert_eq!(result.data, vec![4.0, 4.0, 10.0, 8.0]);
    }

    #[test]
    fn test_scalar_mul() {
        let m = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = 2.0 * &m;
        assert_eq!(result.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_div() {
        let m = Mat::from_vec(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        let result = &m / 2.0;
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_neg() {
        let m = Mat::from_vec(2, 2, vec![1.0, -2.0, 3.0, -4.0]);
        let result = -&m;
        assert_eq!(result.data, vec![-1.0, 2.0, -3.0, 4.0]);
    }
}
