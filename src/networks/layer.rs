use ndarray::{Array, Array2, ArrayD, Axis, Ix1, Ix2};
use ndarray_rand::{RandomExt, rand};

use crate::params::{
  initializer::Initializer, param::LearnableParameter, regularizer::Regularizer,
};

pub trait Layer {
  fn forward(&mut self, input: ArrayD<f64>) -> ArrayD<f64>;
  fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64>;
  fn params_mut(&mut self) -> Vec<&mut LearnableParameter>;
  fn set_training(&mut self, training: bool);
}

pub struct AffineLayer {
  weight: LearnableParameter,
  bias: LearnableParameter,
  input_cache: Option<Array<f64, Ix2>>,
  weight_reg: Option<Box<dyn Regularizer>>,
  training: bool,
}

impl AffineLayer {
  pub fn new<O, P>(
    input_dim: usize,
    output_dim: usize,
    weight_init: &O,
    bias_init: &P,
    weight_reg: Option<Box<dyn Regularizer>>,
  ) -> Self
  where
    O: Initializer,
    P: Initializer,
  {
    let weight = LearnableParameter::new(&[input_dim, output_dim], weight_init);
    let bias = LearnableParameter::new(&[output_dim], bias_init);
    AffineLayer {
      weight,
      bias,
      input_cache: None,
      weight_reg,
      training: true,
    }
  }
}

impl Layer for AffineLayer {
  fn forward(&mut self, input: ArrayD<f64>) -> ArrayD<f64> {
    let x = input.view().into_dimensionality::<Ix2>().unwrap();
    self.input_cache = Some(x.to_owned());
    let w = self
      .weight
      .value
      .view()
      .into_dimensionality::<Ix2>()
      .unwrap();
    let b = self.bias.value.view().into_dimensionality::<Ix1>().unwrap();
    (x.dot(&w) + b.broadcast((x.shape()[0], b.len())).unwrap()).into_dyn()
  }

  fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
    let x = self.input_cache.as_ref().expect("Input cache is not set");
    let w = self
      .weight
      .value
      .view()
      .into_dimensionality::<Ix2>()
      .unwrap();
    let dy = grad.view().into_dimensionality::<Ix2>().unwrap();
    self.weight.grads = match &self.weight_reg {
      Some(reg) => reg.apply(
        x.t().dot(&dy).into_dyn(),
        self.weight.value.clone().into_dyn(),
      ),
      _ => x.t().dot(&dy).into_dyn(),
    };
    self.bias.grads = dy.sum_axis(Axis(0)).into_dyn();
    dy.dot(&w.t()).into_dyn()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter> {
    vec![&mut self.weight, &mut self.bias]
  }
  fn set_training(&mut self, training: bool) {
    self.training = training;
  }
}

pub struct ReLU {
  input_cache: Option<Array<f64, Ix2>>,
}

impl ReLU {
  pub fn new() -> Self {
    ReLU { input_cache: None }
  }
}

impl Layer for ReLU {
  fn forward(&mut self, input: ArrayD<f64>) -> ArrayD<f64> {
    let x = input.view().into_dimensionality::<Ix2>().unwrap();
    self.input_cache = Some(x.to_owned());
    x.mapv(|x| if x > 0.0 { x } else { 0.0 }).into_dyn()
  }

  fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
    let grad = grad.view().into_dimensionality::<Ix2>().unwrap();
    (grad.to_owned()
      * self
        .input_cache
        .as_ref()
        .expect("Input cache is not set")
        .mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
        .to_owned())
    .into_dyn()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter> {
    vec![]
  }
  fn set_training(&mut self, _: bool) {}
}

pub struct Softmax {
  output_cache: Array<f64, Ix2>,
}

impl Softmax {
  pub fn new() -> Self {
    Softmax {
      output_cache: Array::zeros((0, 0)),
    }
  }
}

impl Layer for Softmax {
  fn forward(&mut self, input: ArrayD<f64>) -> ArrayD<f64> {
    let x = input.view().into_dimensionality::<Ix2>().unwrap();
    let mut output = x.to_owned();
    for mut row in output.axis_iter_mut(Axis(0)) {
      let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
      row.mapv_inplace(|x: f64| (x - max).exp());
      let sum = row.sum();
      row.mapv_inplace(|x: f64| x / sum);
    }
    self.output_cache = output.clone();
    output.into_dyn()
  }

  fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
    let dy = grad.view().into_dimensionality::<Ix2>().unwrap();
    let mut result = Array2::<f64>::zeros(dy.raw_dim());
    for ((mut out_row, dyi), y) in result
      .axis_iter_mut(Axis(0))
      .zip(dy.axis_iter(Axis(0)))
      .zip(self.output_cache.axis_iter(Axis(0)))
    {
      let y_vec = y.insert_axis(Axis(0));
      let dyi_vec = dyi.insert_axis(Axis(0));
      let temp = Array2::from_diag(&y) - y_vec.t().dot(&y_vec);
      let dot_result = dyi_vec.dot(&temp);
      let row = &dot_result.row(0);
      out_row.assign(&row);
    }
    result.into_dyn()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter> {
    vec![]
  }
  fn set_training(&mut self, _: bool) {}
}

pub struct BatchNorm {
  weight: LearnableParameter,
  bias: LearnableParameter,
  mean: Array<f64, Ix1>,
  var: Array<f64, Ix1>,
  z_cache: Option<Array<f64, Ix2>>,
  training: bool,
}

impl BatchNorm {
  pub fn new<O, P>(input_dim: usize, weight_init: &O, bias_init: &P) -> Self
  where
    O: Initializer,
    P: Initializer,
  {
    let weight = LearnableParameter::new(&[input_dim], weight_init);
    let bias = LearnableParameter::new(&[input_dim], bias_init);
    let mean = Array::zeros(input_dim);
    let var = Array::zeros(input_dim);
    BatchNorm {
      weight,
      bias,
      mean,
      var,
      z_cache: None,
      training: true,
    }
  }
}

impl Layer for BatchNorm {
  fn forward(&mut self, input: ArrayD<f64>) -> ArrayD<f64> {
    let x = input.view().into_dimensionality::<Ix2>().unwrap();
    let mean = x.mean_axis(Axis(0)).unwrap();
    let var = x.var_axis(Axis(0), 0.0);
    let std = var.mapv(|x| x.max(1e-15).sqrt());
    let z = (x.to_owned() - mean.broadcast((x.shape()[0], mean.shape()[0])).unwrap()) / &std;
    let y = &z
      * self
        .weight
        .value
        .view()
        .into_dimensionality::<Ix1>()
        .unwrap()
        .to_owned()
      + self.bias.value.view().into_dimensionality::<Ix1>().unwrap();
    self.z_cache = Some(z);
    self.mean = mean;
    self.var = var;
    y.into_dyn()
  }
  fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
    let dy = grad.view().into_dimensionality::<Ix2>().unwrap();
    let z = self.z_cache.as_ref().expect("Z cache is not set");
    let dw = Array2::from_shape_vec(
      (dy.shape()[0], z.shape()[1]),
      dy.axis_iter(Axis(0))
        .zip(z.axis_iter(Axis(0)))
        .flat_map(|(dy, z)| &dy * &z)
        .collect(),
    )
    .unwrap()
    .sum_axis(Axis(0));
    self.weight.grads = dw.clone().into_dyn();
    let db = dy.sum_axis(Axis(0));
    self.bias.grads = db.clone().into_dyn();
    let b = dy.shape()[0] as f64;
    let shape = dy.raw_dim();
    let gamma_over_sigma = self.weight.value.clone() / self.var.mapv(|x| x.max(1e-15).sqrt());
    let coeff = gamma_over_sigma.broadcast(shape).unwrap();
    let dgamma_b = dw.broadcast(shape).unwrap();
    let dbeta_b = db.broadcast(shape).unwrap();
    let z_mul_dgamma = z * &dgamma_b;
    let subtract_term = (&z_mul_dgamma + &dbeta_b) / b;
    (&coeff * &(dy.to_owned() - subtract_term.to_owned())).into_dyn()
  }
  fn params_mut(&mut self) -> Vec<&mut LearnableParameter> {
    vec![&mut self.weight, &mut self.bias]
  }
  fn set_training(&mut self, _: bool) {}
}

pub struct Dropout {
  p: f64,
  mask: Option<Array<f64, Ix2>>,
  training: bool,
}
impl Dropout {
  pub fn new(p: f64) -> Self {
    Dropout {
      p,
      mask: None,
      training: true,
    }
  }
}
impl Layer for Dropout {
  fn forward(&mut self, input: ArrayD<f64>) -> ArrayD<f64> {
    if !self.training {
      return input;
    }
    let x = input.view().into_dimensionality::<Ix2>().unwrap();
    let mask = Array::random(x.raw_dim(), rand::distributions::Uniform::new(0.0, 1.0));
    self.mask = Some(mask.clone());
    let output = &x * mask.mapv(|x| if x < self.p { 0.0 } else { 1.0 });
    let output = output / (1.0 - self.p);
    output.into_dyn()
  }

  fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
    let dy = grad.view().into_dimensionality::<Ix2>().unwrap();
    let mask = self.mask.as_ref().expect("Mask is not set");
    (dy.to_owned() * mask / (1.0 - self.p)).into_dyn()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter> {
    vec![]
  }
  fn set_training(&mut self, training: bool) {
    self.training = training;
  }
}
