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

  fn backward(&mut self, grad: GPUTensor<f64, I>) -> GPUTensor<f64, I> {
    let mut output = grad;
    for layer in self.layers.iter_mut().rev() {
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

pub struct DimensionConverter<I, O>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
  O: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  upstream: Box<dyn Layer<I, I>>,
  downstream: Box<dyn Layer<O, O>>,
  converter: Box<dyn Layer<I, O>>,
}

impl<I, O> DimensionConverter<I, O>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
  O: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  pub fn new(
    upstream: Box<dyn Layer<I, I>>,
    converter: Box<dyn Layer<I, O>>,
    downstream: Box<dyn Layer<O, O>>,
  ) -> Self {
    DimensionConverter {
      upstream,
      converter,
      downstream,
    }
  }
}

impl<I, O> Layer<I, O> for DimensionConverter<I, O>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
  O: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  fn forward(&mut self, input: GPUTensor<f64, I>) -> GPUTensor<f64, O> {
    let upstream_output = self.upstream.forward(input);
    let converted_output = self.converter.forward(upstream_output);
    self.downstream.forward(converted_output)
  }

  fn backward(&mut self, grad: GPUTensor<f64, O>) -> GPUTensor<f64, I> {
    let downstream_grad = self.downstream.backward(grad);
    let converted_grad = self.converter.backward(downstream_grad);
    self.upstream.backward(converted_grad)
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    let mut params = self.upstream.params_mut();
    params.extend(self.converter.params_mut());
    params.extend(self.downstream.params_mut());
    params
  }

  fn set_training(&mut self, training: bool) {
    self.upstream.set_training(training);
    self.converter.set_training(training);
    self.downstream.set_training(training);
  }
}
