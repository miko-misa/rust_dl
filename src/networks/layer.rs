use crate::{
  clarray::{
    env::env,
    tensor::{DynamicGPUTensor, GPUMatrix, GPUTensor, GPUVector, Tensor},
  },
  params::{
    initializer::{Initializer, ZeroInitializer},
    param::LearnableParameter,
  },
};

pub trait Layer<I, O>
where
  I: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
  O: AsRef<[usize]> + AsMut<[usize]> + std::fmt::Debug + Clone,
{
  fn forward(&mut self, input: GPUTensor<f64, I>) -> GPUTensor<f64, O>;
  fn backward(&mut self, grad: GPUTensor<f64, O>) -> GPUTensor<f64, I>;
  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>>;
  fn set_training(&mut self, training: bool);
}

pub struct AffineLayer {
  weight: LearnableParameter<f64>,
  bias: LearnableParameter<f64>,
  training: bool,
  input_cache: Option<GPUTensor<f64, [usize; 2]>>,
}

impl AffineLayer {
  pub fn new<O>(input_dim: usize, output_dim: usize, weight_init: &O) -> Self
  where
    O: Initializer<f64, [usize; 2]>,
  {
    let weight = LearnableParameter::new([input_dim, output_dim], weight_init);
    let bias = LearnableParameter::new([output_dim], &ZeroInitializer);
    AffineLayer {
      weight,
      bias,
      training: true,
      input_cache: None,
    }
  }
}

impl Layer<[usize; 2], [usize; 2]> for AffineLayer {
  fn forward(&mut self, input: GPUMatrix<f64>) -> GPUMatrix<f64> {
    if self.training {
      self.input_cache = Some(input.clone());
    }
    let weight_tensor = self
      .weight
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 2]>>()
      .expect("Failed to downcast weight to GPUMatrix");
    let bias_tensor = self
      .bias
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 1]>>()
      .expect("Failed to downcast bias to GPUVector");
    let output = &input.dot(weight_tensor).unwrap();
    let output = (output + &bias_tensor.broadcast_matrix(input.shape[0]).unwrap()).unwrap();
    output
  }

  fn backward(&mut self, grad: GPUTensor<f64, [usize; 2]>) -> GPUTensor<f64, [usize; 2]> {
    let x = self
      .input_cache
      .as_ref()
      .expect("Input cache is not set. Ensure forward is called before backward.");
    let w = self
      .weight
      .value
      .as_any()
      .downcast_ref::<GPUTensor<f64, [usize; 2]>>()
      .expect("Failed to downcast weight to GPUMatrix");
    self.weight.grads = Box::new(x.t().dot(&grad).unwrap());
    let bias_grad = grad.t().row_sum().unwrap();
    self.bias.grads = Box::new(bias_grad);
    grad.dot(&w.t()).unwrap()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![&mut self.weight, &mut self.bias]
  }

  fn set_training(&mut self, training: bool) {
    self.training = training;
  }
}

pub struct ReLU {
  input_cache: Option<GPUTensor<f64, [usize; 2]>>,
}

impl ReLU {
  pub fn new() -> Self {
    ReLU { input_cache: None }
  }
}

impl Layer<[usize; 2], [usize; 2]> for ReLU {
  fn forward(&mut self, input: GPUMatrix<f64>) -> GPUMatrix<f64> {
    self.input_cache = Some(input.clone());
    input.mapv(|x| if x > 0.0 { x } else { 0.0 }).unwrap()
  }

  fn backward(&mut self, grad: GPUTensor<f64, [usize; 2]>) -> GPUTensor<f64, [usize; 2]> {
    let mask = self
      .input_cache
      .as_ref()
      .expect("Input cache is not set. Ensure forward is called before backward.")
      .mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
      .unwrap();
    (&grad * &mask).unwrap()
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![]
  }

  fn set_training(&mut self, _training: bool) {}
}

pub struct Softmax {
  outpu_cache: Option<GPUTensor<f64, [usize; 2]>>,
}

impl Softmax {
  pub fn new() -> Self {
    Softmax { outpu_cache: None }
  }
}

impl Layer<[usize; 2], [usize; 2]> for Softmax {
  fn forward(&mut self, input: GPUMatrix<f64>) -> GPUMatrix<f64> {
    let data = input.to_cpu().unwrap().data;
    // println!("Input data: {:?}", data[0]);
    let max_val = data
      .into_iter()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap();
    let exp_input = (&input - max_val).unwrap().mapv(|x| x.exp()).unwrap();
    let sum_exp = exp_input
      .row_sum()
      .unwrap()
      .broadcast_matrix(input.shape[1])
      .unwrap()
      .t()
      .clip(1e-12, f64::MAX)
      .unwrap();
    let output = (&exp_input / &sum_exp).unwrap();
    self.outpu_cache = Some(output.clone());
    output.contiguous().unwrap()
  }

  fn backward(&mut self, grad: GPUTensor<f64, [usize; 2]>) -> GPUTensor<f64, [usize; 2]> {
    /*
    let y = self
      .outpu_cache
      .as_ref()
      .expect("Input cache is not set. Ensure forward is called before backward.");
    let dx = GPUMatrix::zeros(y.shape, env()).unwrap();
    for ((dxi, dyi), yi) in dx.row_iter().zip(grad.row_iter()).zip(y.row_iter()) {
      let diag_y = GPUMatrix::from_diag(&yi, env()).unwrap();
      let yi_mat = yi.broadcast_matrix(1).unwrap();
      let dyi_mat = dyi.broadcast_matrix(1).unwrap();
      let out = dyi_mat
        .dot(&(&diag_y - &yi_mat.t().dot(&yi_mat).unwrap()).unwrap())
        .unwrap();
      let _ = dxi.write(&out.row_iter().nth(0).expect("Iterator is empty"));
    }
    dx
    */
    grad
  }

  fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
    vec![]
  }

  fn set_training(&mut self, _training: bool) {}
}
