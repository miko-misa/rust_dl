use ndarray::{Array, ArrayD, IxDyn};
use ndarray_rand::{RandomExt, rand_distr::Normal};

use super::initializer::{self, Initializer};

pub struct LearnableParameter {
  pub id: usize,
  pub value: ArrayD<f64>,
  pub grads: ArrayD<f64>,
}

impl LearnableParameter {
  pub fn new<O>(shape: &[usize], initializer: &O) -> Self
  where
    O: Initializer,
  {
    initializer.initialize(&shape)
  }
}
