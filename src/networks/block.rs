use ndarray::ArrayD;

use crate::params::param::LearnableParameter;

use super::layer::Layer;

pub struct Sequential {
  layers: Vec<Box<dyn Layer>>,
}
impl Sequential {
  pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
    Sequential { layers }
  }
}

impl Layer for Sequential {
  fn forward(&mut self, input: ArrayD<f64>) -> ArrayD<f64> {
    let mut output = input;
    for layer in &mut self.layers {
      output = layer.forward(output);
    }
    output
  }
  fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
    let mut output = grad;
    for layer in self.layers.iter_mut().rev() {
      output = layer.backward(output);
    }
    output
  }
  fn params_mut(&mut self) -> Vec<&mut LearnableParameter> {
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
