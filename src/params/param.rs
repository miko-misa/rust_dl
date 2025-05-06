use ndarray::ArrayD;

use super::initializer::Initializer;

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
