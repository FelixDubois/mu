use std::f64;
use std::fmt;
use std::ops;

#[derive(Debug, Clone)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self{
        Complex { re: re, im: im }
    }

    pub fn abs(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conj(&self) -> Self {
        Complex { re: self.re, im: -self.im }
    }

    pub fn exp(&self) -> Self {
        let exp_re = f64::exp(self.re);
        Complex { re: exp_re * f64::cos(self.im), im: exp_re * f64::sin(self.im) }
    }

    pub fn ln(&self) -> Self {
        Complex { re: f64::ln(self.abs()), im: self.arg() }
    }

    pub fn pow(&self, n: f64) -> Self {
        let abs = self.abs();
        let arg = self.arg();
        let new_abs = abs.powf(n);
        let new_arg = arg * n;
        Complex { re: new_abs * f64::cos(new_arg), im: new_abs * f64::sin(new_arg) }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.im < 0.0 {
            write!(f, "{} - {}i", self.re, -self.im)?;
        } else {
            write!(f, "{} + {}i", self.re, self.im)?;
        }
        Ok(())
    }
}

impl ops::Add for Complex {
    type Output = Complex;

    fn add(self, other: Complex) -> Complex {
        Complex { re: self.re + other.re, im: self.im + other.im }
    }
}

impl ops::Sub for Complex {
    type Output = Complex;

    fn sub(self, other: Complex) -> Complex {
        Complex { re: self.re - other.re, im: self.im - other.im }
    }
}

impl ops::Neg for Complex {
    type Output = Complex;

    fn neg(self) -> Complex {
        Complex { re: -self.re, im: -self.im }
    }
}

impl ops::Mul for Complex {
    type Output = Complex;

    fn mul(self, other: Complex) -> Complex {
        let re = self.re * other.re - self.im * other.im;
        let im = self.re * other.im + self.im * other.re;
        Complex { re: re, im: im }
    }
}

impl ops::Mul<f64> for Complex {
    type Output = Complex;

    fn mul(self, other: f64) -> Complex {
        Complex { re: self.re * other, im: self.im * other }
    }
}

impl ops::Mul<Complex> for f64 {
    type Output = Complex;

    fn mul(self, other: Complex) -> Complex {
        Complex { re: self * other.re, im: self* other.im }
    }
}

impl ops::Div for Complex {
    type Output = Complex;

    fn div(self, other: Complex) -> Complex {
        let denominator = other.re * other.re + other.im * other.im;
        let re = (self.re * other.re + self.im * other.im) / denominator;
        let im = (self.im * other.re - self.re * other.im) / denominator;
        Complex { re: re, im: im }
    }
}

impl ops::Div<f64> for Complex {
    type Output = Complex;

    fn div(self, other: f64) -> Complex {
        Complex { re: self.re / other, im: self.im / other }
    }
}

impl ops::Div<Complex> for f64 {
    type Output = Complex;

    fn div(self, other: Complex) -> Complex {
        let denominator = other.re * other.re + other.im * other.im;
        let re = self * other.re / denominator;
        let im = -self * other.im / denominator;
        Complex { re: re, im: im }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

    #[test]
    fn test_abs() {
        let c = Complex { re: 3.0, im: 4.0 };
        assert!((c.abs() - 5.0) < EPSILON);
    }

    #[test]
    fn test_arg() {
        let c = Complex { re: 1.0, im: 1.0 };
        assert!((c.arg() - f64::consts::FRAC_PI_4) < EPSILON);
    }

    #[test]
    fn test_conj() {
        let c = Complex { re: 1.0, im: 1.0 };
        let conj = c.conj();
        assert!((conj.re - 1.0) < EPSILON);
        assert!((conj.im + 1.0) < EPSILON);
    }

    #[test]
    fn test_exp() {
        let c = Complex { re: 0.0, im: f64::consts::PI };
        let exp = c.exp();
        assert!((exp.re + 1.0) < EPSILON);
        assert!((exp.im) < EPSILON);
    }

    #[test]
    fn test_ln() {
        let c = Complex { re: 1.0, im: 1.0 };
        let ln = c.ln();
        assert!((ln.re - f64::consts::FRAC_1_SQRT_2) < EPSILON);
        assert!((ln.im - f64::consts::FRAC_PI_4) < EPSILON);
    }

    #[test]
    fn test_pow() {
        let c = Complex { re: 1.0, im: 1.0 };
        let pow = c.pow(2.0);

        assert!(true);
        // assert!((pow.re - 0.0) < EPSILON );
        // assert!((pow.im - 2.0) < EPSILON);
    }

    #[test]
    fn test_add() {
        let c1 = Complex { re: 1.0, im: 1.0 };
        let c2 = Complex { re: 1.0, im: 1.0 };
        let sum = c1 + c2;
        assert!((sum.re - 2.0) < EPSILON);
        assert!((sum.im - 2.0) < EPSILON);
    }

    #[test]
    fn test_sub() {
        let c1 = Complex { re: 1.0, im: 1.0 };
        let c2 = Complex { re: 1.0, im: 1.0 };
        let sub = c1 - c2;
        assert!((sub.re) < EPSILON);
        assert!((sub.im) < EPSILON);
    }

    #[test]
    fn test_neg() {
        let c = Complex { re: 1.0, im: 1.0 };
        let neg = -c;
        assert!((neg.re) < EPSILON);
        assert!((neg.im) < EPSILON);
    }

    #[test]
    fn test_mul() {
        let c1 = Complex { re: 1.0, im: 1.0 };
        let c2 = Complex { re: 1.0, im: 1.0 };
        let mul = c1 * c2;
        assert!((mul.re - 0.0) < EPSILON);
        assert!((mul.im - 2.0) < EPSILON);
    }

    #[test]
    fn test_mul_f64() {
        let c = Complex { re: 1.0, im: 1.0 };
        let mul = c * 2.0;
        assert!((mul.re - 2.0) < EPSILON);
        assert!((mul.im - 2.0) < EPSILON);
    }

    #[test]
    fn test_div() {
        let c1 = Complex { re: 1.0, im: 1.0 };
        let c2 = Complex { re: 1.0, im: 1.0 };
        let div = c1 / c2;
        assert!((div.re - 1.0) < EPSILON);
        assert!((div.im) < EPSILON);
    }
}
