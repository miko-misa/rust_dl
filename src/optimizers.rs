use crate::networks::layer::LearnableParameter;

pub trait Optimizer {
  fn update(&mut self, params: Vec<&mut LearnableParameter>);
}

pub struct SGD {
  learning_rate: f64,
}

impl SGD {
  pub fn new(learning_rate: f64) -> Self {
    SGD { learning_rate }
  }
}
impl Optimizer for SGD {
  fn update(&mut self, params: Vec<&mut LearnableParameter>) {
    for param in params {
      param.value -= &(self.learning_rate * &param.grads);
      param.grads.fill(0.0); // Reset gradients after update
    }
  }
}
