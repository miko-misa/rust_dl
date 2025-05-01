use ndarray::{Array, Array2, ArrayD, Axis, Ix1, Ix2, IxDyn, Zip, arr1};
use ndarray_rand::{
  RandomExt,
  rand_distr::{Normal, num_traits::Float},
};
use uuid::Uuid;

fn generate_normal_array(shape: &[usize]) -> ArrayD<f64> {
  // 正規分布のインスタンスを作成
  let normal = Normal::new(0.0, 1.0 / (shape[0] as f64).sqrt()).unwrap();
  // IxDynで動的次元を指定し、配列を生成
  Array::random(IxDyn(shape), normal)
}

pub struct LearnableParameter {
  pub id: usize,
  pub value: ArrayD<f64>,
  pub grads: ArrayD<f64>,
}

pub trait Layer {
  fn forward(&mut self, input: ArrayD<f64>) -> ArrayD<f64>;
  fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64>;
  fn params_mut(&mut self) -> Vec<&mut LearnableParameter>;
}

pub struct AffineLayer {
  weight: LearnableParameter,
  bias: LearnableParameter,
  input_cache: Option<Array<f64, Ix2>>,
}

impl AffineLayer {
  pub fn new(input_dim: usize, output_dim: usize) -> Self {
    let weight = LearnableParameter {
      id: Uuid::new_v4().as_u128() as usize,
      value: generate_normal_array(&[input_dim, output_dim]),
      grads: Array::zeros((input_dim, output_dim)).into_dyn(),
    };
    let bias = LearnableParameter {
      id: Uuid::new_v4().as_u128() as usize,
      value: generate_normal_array(&[output_dim]),
      grads: Array::zeros(output_dim).into_dyn(),
    };
    AffineLayer {
      weight,
      bias,
      input_cache: None,
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
    self.weight.grads = x.t().dot(&dy).into_dyn();
    self.bias.grads = dy.sum_axis(Axis(0)).into_dyn();
    dy.dot(&w.t()).into_dyn()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter> {
    vec![&mut self.weight, &mut self.bias]
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
}
