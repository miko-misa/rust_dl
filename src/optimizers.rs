use ndarray::ArrayD;
use std::collections::HashMap;

use crate::params::param::LearnableParameter;

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
    }
  }
}

pub struct Momentum {
  learning_rate: f64,
  momentum: f64,
  velocity: HashMap<usize, ArrayD<f64>>,
}
impl Momentum {
  pub fn new(learning_rate: f64, momentum: f64) -> Self {
    Momentum {
      learning_rate,
      momentum,
      velocity: HashMap::new(),
    }
  }
}
impl Optimizer for Momentum {
  fn update(&mut self, params: Vec<&mut LearnableParameter>) {
    for param in params {
      let m = self
        .velocity
        .entry(param.id)
        .or_insert_with(|| ArrayD::zeros(param.value.shape()));
      *m = self.momentum * &*m - &(self.learning_rate * &param.grads);
      param.value += &*m;
    }
  }
}

pub struct RMSProp {
  learning_rate: f64,
  decay_rate: f64,
  cache: HashMap<usize, ArrayD<f64>>,
}
impl RMSProp {
  pub fn new(learning_rate: f64, decay_rate: f64) -> Self {
    RMSProp {
      learning_rate,
      decay_rate,
      cache: HashMap::new(),
    }
  }
}
impl Optimizer for RMSProp {
  fn update(&mut self, params: Vec<&mut LearnableParameter>) {
    for param in params {
      let cache = self
        .cache
        .entry(param.id)
        .or_insert_with(|| ArrayD::zeros(param.value.shape()));
      *cache =
        self.decay_rate * &*cache + (1.0 - self.decay_rate) * &param.grads.mapv(|x| x.powi(2));
      param.value -= &(self.learning_rate * &param.grads / cache.mapv(|x| x.max(1e-15).sqrt()));
    }
  }
}

pub struct Adam {
  learning_rate: f64,
  beta1: f64,
  beta2: f64,
  m: HashMap<usize, ArrayD<f64>>,
  v: HashMap<usize, ArrayD<f64>>,
  t: usize,
}
impl Adam {
  pub fn new(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
    Adam {
      learning_rate,
      beta1,
      beta2,
      m: HashMap::new(),
      v: HashMap::new(),
      t: 0,
    }
  }
}

impl Optimizer for Adam {
  fn update(&mut self, params: Vec<&mut LearnableParameter>) {
    for param in params {
      let m = self
        .m
        .entry(param.id)
        .or_insert_with(|| ArrayD::zeros(param.value.shape()));
      let v = self
        .v
        .entry(param.id)
        .or_insert_with(|| ArrayD::zeros(param.value.shape()));
      *m = self.beta1 * &*m + (1.0 - self.beta1) * &param.grads;
      *v = self.beta2 * &*v + (1.0 - self.beta2) * &param.grads.mapv(|x| x.powi(2));
      let m_hat = m.clone() / (1.0 - self.beta1.powi(self.t as i32 + 1));
      let v_hat = v.clone() / (1.0 - self.beta2.powi(self.t as i32 + 1));
      param.value -= &(self.learning_rate * m_hat / (v_hat.mapv(|x| x.max(1e-15).sqrt())));
      self.t += 1;
    }
  }
}

pub struct AdamW {
  learning_rate: f64,
  beta1: f64,
  beta2: f64,
  weight_decay: f64,
  m: HashMap<usize, ArrayD<f64>>,
  v: HashMap<usize, ArrayD<f64>>,
  t: usize,
}

impl AdamW {
  pub fn new(learning_rate: f64, beta1: f64, beta2: f64, weight_decay: f64) -> Self {
    AdamW {
      learning_rate,
      beta1,
      beta2,
      weight_decay,
      m: HashMap::new(),
      v: HashMap::new(),
      t: 0,
    }
  }
}

impl Optimizer for AdamW {
  fn update(&mut self, params: Vec<&mut LearnableParameter>) {
    for param in params {
      let m = self
        .m
        .entry(param.id)
        .or_insert_with(|| ArrayD::zeros(param.value.shape()));
      let v = self
        .v
        .entry(param.id)
        .or_insert_with(|| ArrayD::zeros(param.value.shape()));
      *m = self.beta1 * &*m + (1.0 - self.beta1) * &param.grads;
      *v = self.beta2 * &*v + (1.0 - self.beta2) * &param.grads.mapv(|x| x.powi(2));
      let m_hat = m.clone() / (1.0 - self.beta1.powi(self.t as i32 + 1));
      let v_hat = v.clone() / (1.0 - self.beta2.powi(self.t as i32 + 1));
      param.value -= &(self.learning_rate * m_hat / (v_hat.mapv(|x| x.max(1e-15).sqrt())));
      param.value -= &(self.weight_decay * &param.value);
      self.t += 1;
    }
  }
}
