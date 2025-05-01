use ndarray::{Array, ArrayD, IxDyn};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use uuid::Uuid;

use super::param::LearnableParameter;

pub trait Initializer {
  fn initialize(&self, shape: &[usize]) -> LearnableParameter;
}

pub struct ZeroInitializer;
impl Initializer for ZeroInitializer {
  fn initialize(&self, shape: &[usize]) -> LearnableParameter {
    LearnableParameter {
      id: Uuid::new_v4().as_u128() as usize,
      value: Array::zeros(IxDyn(shape)),
      grads: Array::zeros(IxDyn(shape)),
    }
  }
}

pub struct XavierInitializer;
impl Initializer for XavierInitializer {
  fn initialize(&self, shape: &[usize]) -> LearnableParameter {
    let normal = Normal::new(0.0, 1.0 / (shape[0] as f64).sqrt()).unwrap();
    LearnableParameter {
      id: Uuid::new_v4().as_u128() as usize,
      value: Array::random(IxDyn(shape), normal),
      grads: Array::zeros(IxDyn(shape)),
    }
  }
}

pub struct HeInitializer;
impl Initializer for HeInitializer {
  fn initialize(&self, shape: &[usize]) -> LearnableParameter {
    let normal = Normal::new(0.0, (2.0 / shape[0] as f64).sqrt()).unwrap();
    LearnableParameter {
      id: Uuid::new_v4().as_u128() as usize,
      value: Array::random(IxDyn(shape), normal),
      grads: Array::zeros(IxDyn(shape)),
    }
  }
}
