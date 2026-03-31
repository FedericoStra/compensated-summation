/*! This crate implements summation algorithms that significantly reduce the numerical error in the total obtained by adding a sequence of finite-precision floating-point numbers, compared to the obvious approach.

# Algorithms

#### Exact addition

[`two_sum()`] and [`fast_two_sum()`] allow to compute the rounded addition of two floating-point numbers and the associated numerical error.

Both functions return a tuple `(s, t)` where `s` is the floating-point sum rounded to nearest and `t` is the floating-point error.

#### Compensated summation

[`KahanBabuska`] and [`KahanBabuskaNeumaier`] allow to compute compensated sums using the [Kahan-Babuška](https://en.wikipedia.org/wiki/Kahan_summation_algorithm#The_algorithm) and [Kahan-Babuška-Neumaier](https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements) algorithms respectively.

Both types are generic over a parameter `T: num_traits::float::Float`, which is usually [`f32`] or [`f64`] and can typically be inferred.

They support addition and subtraction (also with assignment) of `T` and `&T`.
The estimated total sum (of type `T`) can be retrieved with a method called `total()`.

Both types also implement [`core::iter::Sum`], which means that iterators of floating-point numbers can be conveniently summed.

# Examples

An empty accumulator for the Kahan-Babuška-Neumaier algorithm can be created with [`KahanBabuskaNeumaier::new()`];
floating-point numbers can be added to it and also subtracted from it;
finally, the estimated total sum can be retrieved with the [`KahanBabuskaNeumaier::total()`] method.

```
use compensated_summation::KahanBabuskaNeumaier;
let mut sum = KahanBabuskaNeumaier::new();
sum += 0.1;
sum += 0.2;
sum -= 0.3;
assert_eq!(sum.total(), f64::EPSILON / 8.0);
```

An iterator of floating-point numbers can be conveniently summed via its [`Iterator::sum()`] method, specifying the desired algorithm.

```
use compensated_summation::KahanBabuska;
let iter = [0.1; 10].iter();
assert_eq!(iter.sum::<KahanBabuska<_>>().total(), 1.0);
```

# Caution!

A word of caution: floating-point arithmetic is subtler than one may naively think!

For example, the *exact sum* of the double precision ([`f64`]) floating-point numbers
`0.1`, `0.2` and `-0.3` is **not zero**, not because of some numerical errors in the computation of
arithmetic operations (I'm speaking of their exact mathematical sum, not `0.1 + 0.2 - 0.3`), but
because `0.1`, `0.2` and `-0.3` do not represent what you may think. Mathematically, they are not
$1/10$, $2/10=1/5$ and $-3/10$ respectively, because these are not
[dyadic rational numbers](https://en.wikipedia.org/wiki/Dyadic_rational)
hence cannot be represented exactly as [`f64`] floating-point numbers. Instead, `0.1`, `0.2` and `-0.3` are
$3602879701896397/36028797018963968$, $3602879701896397/18014398509481984$ and
$-5404319552844595/18014398509481984$ respectively, whose sum is $1/36028797018963968=2^{-55}$, which is equal to `f64::EPSILON / 8.0`.

With this in mind, one can realize that in this case Kahan-Babuška-Neumaier computes the correct sum

```
use compensated_summation::*;
assert_eq!([0.1, 0.2, -0.3].iter().sum::<KahanBabuskaNeumaier<f64>>().total(), f64::EPSILON / 8.0);
```

whereas both the naive summation and Kahan-Babuška do not

```
use compensated_summation::*;
assert_eq!([0.1, 0.2, -0.3].iter().sum::<f64>(), f64::EPSILON / 4.0);
assert_eq!([0.1, 0.2, -0.3].iter().sum::<KahanBabuska<f64>>().total(), 0.0);
```

Don't be fooled by the Kahan-Babuška result being `0.0`: from the `f64` perspective it is incorrect.
*/

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![allow(uncommon_codepoints, mixed_script_confusables)]
#![no_std]

use core::iter::Sum;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use num_traits::float::Float;

/// `2Sum` algorithm, see <https://en.wikipedia.org/wiki/2Sum>.
///
/// **Input:** two floating-point numbers $a$ and $b$.
///
/// **Output:** a tuple $(s,t)$ where $s=a\oplus b$ is the floating-point sum [rounded to nearest](https://en.wikipedia.org/wiki/IEEE_754#Roundings_to_nearest) and $t=a+b-(a\oplus b)$ is the floating-point error, so that $a+b=s+t$.
pub fn two_sum<T: Float>(a: T, b: T) -> (T, T) {
    // https://web.archive.org/web/20230122184725/https://ir.cwi.nl/pub/9159/9159D.pdf
    // Equation (4.16) on page 15.
    let s = a + b;
    let aʹ = s - b;
    let bʹ = s - aʹ;
    let δa = a - aʹ;
    let δb = b - bʹ;
    let t = δa + δb;
    (s, t)
}

// This is private, for the time being.
fn two_sub<T: Float>(a: T, b: T) -> (T, T) {
    let s = a - b;
    let aʹ = s + b;
    let bʹ = aʹ - s;
    let δa = a - aʹ;
    let δb = b - bʹ;
    let t = δa - δb;
    (s, t)
}

/// `Fast2Sum` algorithm, see <https://en.wikipedia.org/wiki/2Sum>.
///
/// **Input:** two floating-point numbers $a$ and $b$, of which at least one is zero, or which have normalized exponents $e_a\geq e_b$ (such as when $|a|\geq|b|$).
///
/// **Output:** a tuple $(s,t)$ where $s=a\oplus b$ is the floating-point sum [rounded to nearest](https://en.wikipedia.org/wiki/IEEE_754#Roundings_to_nearest) and $t=a+b-(a\oplus b)$ is the floating-point error, so that $a+b=s+t$.
pub fn fast_two_sum<T: Float>(a: T, b: T) -> (T, T) {
    // https://web.archive.org/web/20220105073915/http://mgnet.org/~douglas/Classes/na-sc/notes/kahan.pdf
    let s = a + b;
    let bʹ = s - a;
    let δb = b - bʹ;
    (s, δb)
}

/// This type is an accumulator for computing a sum with [Kahan-Babuška algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm#The_algorithm).
///
/// The generic parameter `T` should typically implement [`num_traits::float::Float`] and can usually be inferred.
///
/// # Examples
///
/// You can create a new empty accumulator with [`KahanBabuska::new()`];
/// then you can add and subtract floating-point numbers;
/// when you are done, you can retrieve the total with the [`KahanBabuska::total()`] method.
///
/// ```
/// # use compensated_summation::KahanBabuska;
/// let mut sum = KahanBabuska::new();
/// sum += 0.1;
/// sum += 0.2;
/// sum -= 0.3;
/// assert_eq!(sum.total(), 0.0);
/// ```
///
/// In addition, [`KahanBabuska`] implements the [`core::iter::Sum`](#impl-Sum<V>-for-KahanBabuska<T>) trait, which means that an iterator of floating-point numbers can be summed either by calling [`KahanBabuska::sum()`] directly
///
/// ```
/// # use compensated_summation::KahanBabuska;
/// use core::iter::Sum; // remember to import the trait
/// let iter = [0.1, 0.2, -0.3].iter();
/// assert_eq!(KahanBabuska::sum(iter).total(), 0.0);
/// ```
///
/// or by using its [`Iterator::sum()`] method
///
/// ```
/// # use compensated_summation::KahanBabuska;
/// let iter = [0.1, 0.2, -0.3].iter();
/// assert_eq!(iter.sum::<KahanBabuska<_>>().total(), 0.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct KahanBabuska<T> {
    /// Accumulated sum.
    pub sum: T,
    /// Compensation of the error.
    pub comp: T,
}

impl<T: Float> KahanBabuska<T> {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self {
            sum: T::zero(),
            comp: T::zero(),
        }
    }

    /// Get the estimated total sum.
    pub fn total(&self) -> T {
        self.sum + self.comp
    }
}

impl<T: Float> Default for KahanBabuska<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Add<T> for KahanBabuska<T> {
    type Output = Self;
    fn add(mut self, rhs: T) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: Float> AddAssign<T> for KahanBabuska<T> {
    fn add_assign(&mut self, rhs: T) {
        let (s, c) = fast_two_sum(self.sum, rhs + self.comp);
        self.sum = s;
        self.comp = c;
    }
}

impl<T: Float> Sub<T> for KahanBabuska<T> {
    type Output = Self;
    fn sub(mut self, rhs: T) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T: Float> SubAssign<T> for KahanBabuska<T> {
    fn sub_assign(&mut self, rhs: T) {
        let (s, c) = fast_two_sum(self.sum, self.comp - rhs);
        self.sum = s;
        self.comp = c;
    }
}

impl<T: Float> Add<&T> for KahanBabuska<T> {
    type Output = Self;
    fn add(self, rhs: &T) -> Self::Output {
        self + *rhs
    }
}

impl<T: Float> AddAssign<&T> for KahanBabuska<T> {
    fn add_assign(&mut self, rhs: &T) {
        *self += *rhs;
    }
}

impl<T: Float> Sub<&T> for KahanBabuska<T> {
    type Output = Self;
    fn sub(self, rhs: &T) -> Self::Output {
        self - *rhs
    }
}

impl<T: Float> SubAssign<&T> for KahanBabuska<T> {
    fn sub_assign(&mut self, rhs: &T) {
        *self -= *rhs;
    }
}

impl<T: Float, V> Sum<V> for KahanBabuska<T>
where
    Self: AddAssign<V>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = V>,
    {
        // This could be implemented as
        // iter.fold(KahanBabuska::new(), KahanBabuska::add)
        // however, using a for loop improves codegen (smaller assembly).
        let mut sum = KahanBabuska::new();
        for x in iter {
            sum += x;
        }
        sum
    }
}

/// This type is an accumulator for computing a sum with [Kahan-Babuška-Neumaier algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements).
///
/// The generic parameter `T` should typically implement [`num_traits::float::Float`] and can usually be inferred.
///
/// # Examples
///
/// You can create a new empty accumulator with [`KahanBabuskaNeumaier::new()`];
/// then you can add and subtract floating-point numbers;
/// when you are done, you can retrieve the total with the [`KahanBabuskaNeumaier::total()`] method.
///
/// ```
/// # use compensated_summation::KahanBabuskaNeumaier;
/// let mut sum = KahanBabuskaNeumaier::new();
/// sum += 0.1;
/// sum += 0.2;
/// sum -= 0.3;
/// assert_eq!(sum.total(), f64::EPSILON / 8.0);
/// ```
///
/// In addition, [`KahanBabuskaNeumaier`] implements the [`core::iter::Sum`](#impl-Sum<V>-for-KahanBabuskaNeumaier<T>) trait, which means that an iterator of floating-point numbers can be summed either by calling [`KahanBabuskaNeumaier::sum()`] directly
///
/// ```
/// # use compensated_summation::KahanBabuskaNeumaier;
/// use core::iter::Sum; // remember to import the trait
/// let iter = [0.1, 0.2, -0.3].iter();
/// assert_eq!(KahanBabuskaNeumaier::sum(iter).total(), f64::EPSILON / 8.0);
/// ```
///
/// or by using its [`Iterator::sum()`] method
///
/// ```
/// # use compensated_summation::KahanBabuskaNeumaier;
/// let iter = [0.1, 0.2, -0.3].iter();
/// assert_eq!(iter.sum::<KahanBabuskaNeumaier<_>>().total(), f64::EPSILON / 8.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct KahanBabuskaNeumaier<T> {
    /// Accumulated sum.
    pub sum: T,
    /// Compensation of the error.
    pub comp: T,
}

impl<T: Float> KahanBabuskaNeumaier<T> {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self {
            sum: T::zero(),
            comp: T::zero(),
        }
    }

    /// Get the estimated total sum.
    pub fn total(&self) -> T {
        self.sum + self.comp
    }
}

impl<T: Float> Default for KahanBabuskaNeumaier<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Add<T> for KahanBabuskaNeumaier<T> {
    type Output = Self;
    fn add(mut self, rhs: T) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: Float> AddAssign<T> for KahanBabuskaNeumaier<T> {
    fn add_assign(&mut self, rhs: T) {
        let (s, c) = two_sum(self.sum, rhs);
        self.sum = s;
        self.comp = self.comp + c;
    }
}

impl<T: Float> Sub<T> for KahanBabuskaNeumaier<T> {
    type Output = Self;
    fn sub(mut self, rhs: T) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T: Float> SubAssign<T> for KahanBabuskaNeumaier<T> {
    fn sub_assign(&mut self, rhs: T) {
        let (s, c) = two_sub(self.sum, rhs);
        self.sum = s;
        self.comp = self.comp + c;
    }
}

impl<T: Float> Add<&T> for KahanBabuskaNeumaier<T> {
    type Output = Self;
    fn add(self, rhs: &T) -> Self::Output {
        self + *rhs
    }
}

impl<T: Float> AddAssign<&T> for KahanBabuskaNeumaier<T> {
    fn add_assign(&mut self, rhs: &T) {
        *self += *rhs;
    }
}

impl<T: Float> Sub<&T> for KahanBabuskaNeumaier<T> {
    type Output = Self;
    fn sub(self, rhs: &T) -> Self::Output {
        self - *rhs
    }
}

impl<T: Float> SubAssign<&T> for KahanBabuskaNeumaier<T> {
    fn sub_assign(&mut self, rhs: &T) {
        *self -= *rhs;
    }
}

impl<T: Float, V> Sum<V> for KahanBabuskaNeumaier<T>
where
    Self: AddAssign<V>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = V>,
    {
        // This could be implemented as
        // iter.fold(KahanBabuskaNeumaier::new(), KahanBabuskaNeumaier::add)
        // however, using a for loop improves codegen (smaller assembly).
        let mut sum = KahanBabuskaNeumaier::new();
        for x in iter {
            sum += x;
        }
        sum
    }
}

/// Same as [`KahanBabuska`], but with correct spelling of the second surname.
pub type KahanBabuška<T> = KahanBabuska<T>;

/// Same as [`KahanBabuskaNeumaier`], but with correct spelling of the second surname.
pub type KahanBabuškaNeumaier<T> = KahanBabuskaNeumaier<T>;

/// This module is for development purposes only!
///
/// It provides additional functions and alternative implementations used in testing and benchmarking.
///
/// **DO NOT RELY ON IT!**
///
/// Initially this module was gated by `[cfg(test)]`.
/// However, some of these functions turned out to be useful also in benchmarking and
/// to analyze the generated assembly, hence needed a way to be be exported outside of
/// the crate. This is the reason why this module is now public and gated by
/// `#[cfg(any(test, feature = "dev"))]`.
#[cfg(any(test, feature = "dev"))]
pub mod dev;

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_two_sum() {
        use rand::prelude::*;
        use rand_distr::LogNormal;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let dist = LogNormal::new(0.0, 40.0).unwrap();

        for _ in 0..1000 {
            let a = rng.sample(dist);
            let b = rng.sample(dist);
            assert_eq!(two_sum(a, b), dev::abs_two_sum(a, b));
        }
    }

    #[test]
    fn test_two_sub() {
        use rand::prelude::*;
        use rand_distr::LogNormal;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let dist = LogNormal::new(0.0, 40.0).unwrap();

        for _ in 0..1000 {
            let a = rng.sample(dist);
            let b = rng.sample(dist);
            assert_eq!(two_sub(a, b), two_sum(a, -b));
        }
    }

    #[test]
    fn kahan_123() {
        let mut k = KahanBabuska::new();
        let mut s = 0.0;
        k += 0.1;
        s += 0.1;
        k += 0.2;
        s += 0.2;
        k += -0.3;
        s += -0.3;
        assert_eq!(k.sum, 0.0);
        assert_eq!(k.comp, 0.0);
        assert_eq!(s, f64::EPSILON / 4.);
    }

    #[test]
    fn kahan_tenth() {
        let mut k = KahanBabuska::new();
        let mut s = 0.0;
        for _ in 0..10 {
            k += 0.1;
            s += 0.1;
        }
        k += -1.0;
        s += -1.0;
        assert_eq!(k.sum, 0.0);
        assert_eq!(k.comp, 0.0);
        assert_eq!(s, -f64::EPSILON / 2.);
    }

    #[test]
    fn kahan_123_iter() {
        assert_eq!(
            [0.1, 0.2, -0.3].iter().sum::<KahanBabuska<f64>>().total(),
            0.0
        );
        assert_eq!(
            [0.1, 0.2, -0.3]
                .iter()
                .cloned()
                .sum::<KahanBabuska<f64>>()
                .total(),
            0.0
        );
    }

    #[test]
    fn kahan_tenth_iter() {
        assert_eq!([0.1; 10].iter().sum::<KahanBabuska<f64>>().total(), 1.0);
        assert_eq!(
            [0.1; 10].iter().cloned().sum::<KahanBabuska<f64>>().total(),
            1.0
        );
    }

    #[test]
    fn kahan_large() {
        assert_eq!(
            [1.0, 1e100, 1.0, -1e100]
                .iter()
                .sum::<KahanBabuska<f64>>()
                .total(),
            0.0
        );
    }

    #[test]
    fn neumaier_large() {
        assert_eq!(
            [1.0, 1e100, 1.0, -1e100]
                .iter()
                .sum::<KahanBabuskaNeumaier<f64>>()
                .total(),
            2.0
        );
    }

    #[test]
    fn test_correctness() {
        extern crate std;

        use std::vec::Vec;

        use rand::prelude::*;
        use rand_distr::LogNormal;
        use rand_xoshiro::Xoshiro256PlusPlus;

        for seed in 0..100 {
            let rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let values: Vec<f64> = rng
                .sample_iter(LogNormal::new(0.0, 40.0).unwrap())
                .take(1_000)
                .collect();

            assert_eq!(
                values.iter().sum::<KahanBabuska<_>>().total(),
                dev::kahan_babuska_sum(values.iter().cloned())
            );

            assert_eq!(
                values.iter().sum::<KahanBabuskaNeumaier<_>>().total(),
                dev::kahan_babuska_neumaier_sum(values.iter().cloned())
            );
            assert_eq!(
                values.iter().sum::<KahanBabuskaNeumaier<_>>().total(),
                dev::kahan_babuska_neumaier_abs_sum(values.iter().cloned())
            );
            assert_eq!(
                values.iter().sum::<KahanBabuskaNeumaier<_>>().total(),
                dev::kahan_babuska_neumaier_abs_two_sum(values.iter().cloned())
            );
        }
    }
}
