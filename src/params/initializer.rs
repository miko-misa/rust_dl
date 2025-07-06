use num_traits::{FromPrimitive, Num, ToPrimitive};
use ocl::{OclPrm, core::OclNum};
use rand::{rng, thread_rng};
use rand_distr::{Distribution, Normal};

use crate::{
  clarray::env::env,
  clarray::tensor::{GPUTensor, OclComputeNum, Tensor},
  params::param::LearnableParameter,
};
use std::fmt::Debug;

pub trait Initializer<T, D>
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone,
{
  fn initialize(&self, shape: &D) -> LearnableParameter<T>;
}

pub struct ZeroInitializer;
impl<T, D> Initializer<T, D> for ZeroInitializer
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone + 'static,
{
  fn initialize(&self, shape: &D) -> LearnableParameter<T> {
    LearnableParameter {
      id: 0,
      value: Box::new(GPUTensor::zeros(shape.clone(), crate::clarray::env::env().clone()).unwrap()),
      grads: Box::new(GPUTensor::zeros(shape.clone(), crate::clarray::env::env().clone()).unwrap()),
    }
  }
}

pub struct HeInitializer;
impl<T, D> Initializer<T, D> for HeInitializer
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + Debug + Clone + 'static,
{
  fn initialize(&self, shape: &D) -> LearnableParameter<T> {
    let fan_in = shape.as_ref()[0];
    let stddev = (T::from_f64(2.0).unwrap() / T::from_usize(fan_in).unwrap())
      .to_f64()
      .unwrap()
      .sqrt();
    let value_vec = generate_normal_vec(stddev, shape.as_ref().iter().product::<usize>());
    LearnableParameter {
      id: 1,
      value: Box::new(GPUTensor::from_vec(shape.clone(), value_vec, env()).unwrap()),
      grads: Box::new(GPUTensor::zeros(shape.clone(), env()).unwrap()),
    }
  }
}

fn generate_normal_vec<T>(std_dev: f64, n: usize) -> Vec<T>
where
  T: OclComputeNum + FromPrimitive + ToPrimitive + Debug,
{
  let normal_dist = Normal::new(0.0, std_dev.to_f64().unwrap()).unwrap();
  let mut rng = rng();
  normal_dist
    .sample_iter(&mut rng)
    .take(n)
    .filter_map(|x| T::from_f64(x))
    .collect()
}
