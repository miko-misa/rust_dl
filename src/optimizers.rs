use crate::{
  clarray::tensor::{GPUTensor, OclComputeNum},
  params::param::LearnableParameter,
};

pub trait Optimizer<T>
where
  T: OclComputeNum,
{
  fn update(&mut self, params: &mut [LearnableParameter<T>]);
}

pub struct SGD<T>
where
  T: OclComputeNum,
{
  pub learning_rate: T,
}

impl<T> SGD<T>
where
  T: OclComputeNum,
{
  pub fn new(learning_rate: T) -> Self {
    Self { learning_rate }
  }
}

impl<T> Optimizer<T> for SGD<T>
where
  T: OclComputeNum,
{
  fn update(&mut self, params: &mut [LearnableParameter<T>]) {
    for param in params.iter_mut() {
      match param.value.rank() {
        1 => {
          let grads = param
            .grads
            .as_any()
            .downcast_ref::<GPUTensor<T, [usize; 1]>>()
            .unwrap();
          let value_mut = param.value.as_any_mut();
          value_mut
            .downcast_mut::<GPUTensor<T, [usize; 1]>>()
            .unwrap()
            .write(&(grads - &(grads * self.learning_rate).unwrap()).unwrap())
            .unwrap();
        }
        2 => {
          let grads = param
            .grads
            .as_any()
            .downcast_ref::<GPUTensor<T, [usize; 2]>>()
            .unwrap();
          let value_mut = param.value.as_any_mut();
          value_mut
            .downcast_mut::<GPUTensor<T, [usize; 2]>>()
            .unwrap()
            .write(&(grads - &(grads * self.learning_rate).unwrap()).unwrap())
            .unwrap();
        }
        _ => panic!("Unsupported tensor rank for SGD optimizer"),
      }
    }
  }
}
