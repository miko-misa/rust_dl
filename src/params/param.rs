use std::fmt::Debug;

use num_traits::Num;
use ocl::{OclPrm, core::OclNum};

use crate::{
  clarray::tensor::{DynamicGPUTensor, GPUTensor, OclComputeNum},
  params::initializer::Initializer,
};

pub struct LearnableParameter<T>
where
  T: OclComputeNum,
{
  pub id: usize,
  pub value: Box<dyn DynamicGPUTensor<T>>,
  pub grads: Box<dyn DynamicGPUTensor<T>>,
}

impl<T> LearnableParameter<T>
where
  T: OclComputeNum,
{
  pub fn new<O, D>(shape: D, initializer: &O) -> Self
  where
    O: Initializer<T, D>,
    D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
  {
    initializer.initialize(&shape)
  }
}
