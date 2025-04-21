use ndarray::{ArrayBase, Ix2, IxDyn, OwnedRepr};

use super::layer::Layer;

pub trait Block<D = IxDyn> {
  fn forward(&mut self, input: ArrayBase<OwnedRepr<f64>, D>) -> ArrayBase<OwnedRepr<f64>, D>;
  fn backward(&mut self, grad: ArrayBase<OwnedRepr<f64>, D>) -> ArrayBase<OwnedRepr<f64>, D>;
}

pub struct Sequential<D> {
  layers: Vec<Box<dyn Layer<D>>>,
}

impl<D> Sequential<D> {
  pub fn new(layers: Vec<Box<dyn Layer<D>>>) -> Self {
    Self { layers }
  }
}

impl<D> Block<D> for Sequential<D> {
  fn forward(&mut self, input: ArrayBase<OwnedRepr<f64>, D>) -> ArrayBase<OwnedRepr<f64>, D> {
    let mut output = input;
    for layer in &mut self.layers {
      output = layer.forward(output);
    }
    output
  }

  fn backward(&mut self, grad: ArrayBase<OwnedRepr<f64>, D>) -> ArrayBase<OwnedRepr<f64>, D> {
    let mut grad_output = grad;
    for layer in self.layers.iter_mut().rev() {
      grad_output = layer.backward(grad_output);
    }
    grad_output
  }
}
