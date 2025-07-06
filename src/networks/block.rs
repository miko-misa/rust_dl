use crate::{
  clarray::tensor::GPUTensor, networks::layer::Layer, params::param::LearnableParameter,
};

pub struct Sequential<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  pub layers: Vec<Box<dyn Layer<I, I>>>,
}

impl<I> Sequential<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  pub fn new(layers: Vec<Box<dyn Layer<I, I>>>) -> Self {
    Sequential { layers }
  }
}

impl<I> Layer<I, I> for Sequential<I>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  fn forward(&mut self, input: GPUTensor<f64, I>) -> GPUTensor<f64, I> {
    let mut output = input;
    for layer in &mut self.layers {
      output = layer.forward(output);
    }
    output
  }

  fn backward(&self, grad: GPUTensor<f64, I>) -> GPUTensor<f64, I> {
    let mut output = grad;
    for layer in self.layers.iter().rev() {
      output = layer.backward(output);
    }
    output
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    self
      .layers
      .iter_mut()
      .flat_map(|layer| layer.params_mut())
      .collect()
  }

  fn set_training(&mut self, training: bool) {
    for layer in &mut self.layers {
      layer.set_training(training);
    }
  }
}
