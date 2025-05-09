use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::ArrayD;
use ndarray_rand::rand;
use rand::seq::SliceRandom;

use crate::{losses::LossFunction, networks::layer::Layer, optimizers::Optimizer};

pub struct BaseModel {
  main_layer: Box<dyn Layer>,
  loss_func: Box<dyn LossFunction>,
  optimizer: Box<dyn Optimizer>,
}

impl BaseModel {
  pub fn new(
    main_layer: Box<dyn Layer>,
    loss_func: Box<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,
  ) -> Self {
    BaseModel {
      main_layer,
      loss_func,
      optimizer,
    }
  }

  pub fn train(
    &mut self,
    epochs: usize,
    train_data: &Vec<(ArrayD<f64>, ArrayD<f64>)>,
    val_data: &Vec<(ArrayD<f64>, ArrayD<f64>)>,
  ) -> Vec<(f64, f64)> {
    let mut result = vec![];

    let m = MultiProgress::new();

    let total_steps = (epochs * (train_data.len() + val_data.len())) as u64;
    let overall_pb = m.add(ProgressBar::new(total_steps));
    overall_pb.set_style(
      ProgressStyle::default_bar()
        .template("Total Progress: [{bar:40.cyan/blue}] {pos}/{len} ({{elapsed_precise}} ETA:{{eta}}) {{msg}}")
        .unwrap()
        .progress_chars("##-"),
    );
    for epoch in 1..=epochs {
      let mut train_losses = vec![];
      let epoch_pb = m.add(ProgressBar::new(train_data.len() as u64));
      epoch_pb.set_style(
        ProgressStyle::default_bar()
          .template(&format!(
            "[Train] Epoch #{}: [{{bar:40.green/black}}] {{pos}}/{{len}} ({{elapsed_precise}} ETA:{{eta}}) {{msg}}",
            epoch
          ))
          .unwrap()
          .progress_chars("##-"),
      );
      for (x_train, y_train) in train_data {
        let y_pred = self.main_layer.forward(x_train.clone());
        let loss_value = self.loss_func.forward(y_pred.clone(), y_train.clone());
        // println!("Epoch {}: Loss: {}", epoch, loss_value);
        epoch_pb.set_message(format!("Loss: {}", loss_value));
        train_losses.push(loss_value);
        let grad = self.loss_func.backward(y_pred.clone(), y_train.clone());
        self.main_layer.backward(grad);
        self.optimizer.update(self.main_layer.params_mut());
        epoch_pb.inc(1);
        overall_pb.inc(1);
      }
      let mut val_losses = vec![];
      epoch_pb.finish();
      let epoch_pb = m.add(ProgressBar::new(val_data.len() as u64));
      epoch_pb.set_style(
        ProgressStyle::default_bar()
          .template(&format!(
            "[Validation] Epoch #{}: [{{bar:40.green/black}}] {{pos}}/{{len}} ({{elapsed_precise}} ETA:{{eta}}) {{msg}}",
            epoch
          ))
          .unwrap()
          .progress_chars("##-"),
      );
      for (x_val, y_val) in val_data {
        let y_pred = self.main_layer.forward(x_val.clone());
        let loss_value = self.loss_func.forward(y_pred.clone(), y_val.clone());
        // println!("Validation Loss: {}", loss_value);
        epoch_pb.set_message(format!("Loss: {}", loss_value));
        val_losses.push(loss_value);
        epoch_pb.inc(1);
        overall_pb.inc(1);
      }
      epoch_pb.finish();
      let train_loss = train_losses.iter().sum::<f64>() / train_losses.len() as f64;
      let val_loss = val_losses.iter().sum::<f64>() / val_losses.len() as f64;
      result.push((train_loss, val_loss));
    }
    overall_pb.finish_with_message("Total training completed.");
    result
  }

  pub fn train_step(
    &mut self,
    epochs: usize,
    train_data: &Vec<(ArrayD<f64>, ArrayD<f64>)>,
    val_data: &Vec<(ArrayD<f64>, ArrayD<f64>)>,
  ) -> Vec<(f64, f64)> {
    let mut result = vec![];

    let m = MultiProgress::new();

    let total_steps = (epochs * train_data.len()) as u64;
    let overall_pb = m.add(ProgressBar::new(total_steps));
    overall_pb.set_style(
      ProgressStyle::default_bar()
        .template(&format!("Total Progress: [{{bar:40.cyan/blue}}] {{pos}}/{{len}} ({{elapsed_precise}} ETA:{{eta}}) {{msg}}"))
        .unwrap()
        .progress_chars("##-"),
    );

    for epoch in 1..=epochs {
      let epoch_pb = m.add(ProgressBar::new(train_data.len() as u64));
      epoch_pb.set_style(
        ProgressStyle::default_bar()
          .template(&format!(
            "[Train] Epoch #{}: [{{bar:40.green/black}}] {{pos}}/{{len}} ({{elapsed_precise}} ETA:{{eta}}) {{msg}}",
            epoch
          ))
          .unwrap()
          .progress_chars("##-"),
      );

      for (x_train, y_train) in train_data {
        let y_pred = self.main_layer.forward(x_train.clone());
        let loss_value = self.loss_func.forward(y_pred.clone(), y_train.clone());
        let grad = self.loss_func.backward(y_pred.clone(), y_train.clone());
        self.main_layer.backward(grad);
        self.optimizer.update(self.main_layer.params_mut());

        let (x_val, y_val) = val_data.choose(&mut rand::thread_rng()).unwrap();
        let y_pred = self.main_layer.forward(x_val.clone());
        let val_loss = self.loss_func.forward(y_pred.clone(), y_val.clone());
        epoch_pb.set_message(format!(
          "Train Loss: {} Validation Loss: {}",
          loss_value, val_loss
        ));

        result.push((loss_value, val_loss));

        epoch_pb.inc(1);
        overall_pb.inc(1);
      }

      epoch_pb.finish_with_message(format!("Epoch #{} completed.", epoch));
    }
    overall_pb.finish_with_message("Total training completed.");
    result
  }
  pub fn predict(&mut self, input: ArrayD<f64>) -> ArrayD<f64> {
    self.main_layer.forward(input)
  }

  pub fn loss(&mut self, input: ArrayD<f64>, target: ArrayD<f64>) -> f64 {
    let y_pred = self.main_layer.forward(input);
    self.loss_func.forward(y_pred, target)
  }
}
