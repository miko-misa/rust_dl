use crate::clarray::tensor::{GPUTensor, OclComputeNum};

pub trait LossFunction<T, D, E>
where
  T: OclComputeNum,
  D: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
  E: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  fn forward(&mut self, input: GPUTensor<T, D>, target: GPUTensor<T, E>) -> f64;
  fn backward(&self, input: GPUTensor<T, D>, target: GPUTensor<T, E>) -> GPUTensor<T, D>;
}

pub struct CrossEntropyLoss;
impl LossFunction<f64, [usize; 2], [usize; 2]> for CrossEntropyLoss {
  fn forward(
    &mut self,
    input: GPUTensor<f64, [usize; 2]>,
    target: GPUTensor<f64, [usize; 2]>,
  ) -> f64 {
    let esp = 1e-12;
    let input = input.mapv(|x| x.max(esp).min(1.0 - esp).ln()).unwrap();
    let loss = (&input * &target).unwrap().sum().unwrap();
    -loss / input.shape[0] as f64
  }

  fn backward(
    &self,
    input: GPUTensor<f64, [usize; 2]>,
    target: GPUTensor<f64, [usize; 2]>,
  ) -> GPUTensor<f64, [usize; 2]> {
    /*
    let esp = 1e-12;
    let input = input.clip(esp, f64::MAX).unwrap();
    let grad = &(&(&target / &input).unwrap() / (input.shape[0] as f64)).unwrap() * -1.0;
    grad.unwrap()
    */
    let batch_size = input.shape[0] as f64;
    (&(&input - &target).unwrap() / batch_size).unwrap()
  }
}
