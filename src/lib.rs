mod losses;
mod networks;
mod optimizers;
mod params;
mod utils;

#[cfg(test)]
mod tests {
  use crate::losses::{CrossEntropyLoss, LossFunction};
  use crate::networks::layer::{AffineLayer, BatchNorm, Layer, ReLU, Softmax};
  use crate::optimizers::{Adam, Momentum, RMSProp, SGD};
  use crate::params::initializer::{HeInitializer, ZeroInitializer};
  use crate::{
    networks::block::Sequential,
    optimizers::Optimizer,
    utils::{
      self,
      batch::{create_batches, one_hot_encode},
    },
  };
  use ndarray::{ArrayD, IxDyn};
  use ndarray_rand::rand::{self, thread_rng};
  use rand::seq::SliceRandom;
  use utils::load_csv::load_csv_to_ndarray;

  #[test]
  fn it_works() {
    let mnist = load_csv_to_ndarray("data/mnist_train.csv", true).unwrap();
    let mnist_val = load_csv_to_ndarray("data/mnist_test.csv", true).unwrap();
    let he_init = HeInitializer;
    let zero_init = ZeroInitializer;
    let mut model = Sequential::new(vec![
      Box::new(AffineLayer::new(784, 128, &he_init, &zero_init)),
      Box::new(BatchNorm::new(128, &he_init, &zero_init)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(128, 64, &he_init, &zero_init)),
      Box::new(BatchNorm::new(64, &he_init, &zero_init)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(64, 10, &he_init, &zero_init)),
      Box::new(BatchNorm::new(10, &he_init, &zero_init)),
      Box::new(Softmax::new()),
    ]);
    let mut _optimizer_sgd = SGD::new(0.05);
    let mut _optimizer_momentum = Momentum::new(0.05, 0.9);
    let mut _optimizer_rmsprop = RMSProp::new(0.005, 0.9);
    let mut optimizer_adam = Adam::new(0.005, 0.9, 0.999);
    let loss = CrossEntropyLoss::new();
    let mut validate = create_batches(&mnist_val, 0, 4096);
    let mut rng = thread_rng();
    for _ in 0..10 {
      for (x_train, y_train) in create_batches(&mnist, 0, 2048) {
        let y_train = one_hot_encode(&y_train, 10);
        let x_train = x_train / 255.0;
        let y_pred = model.forward(x_train.clone().into_dyn());
        let loss_value = loss.forward(y_pred.clone(), y_train.clone().into_dyn());
        println!("Loss: {}", loss_value);
        let grad = loss.backward(y_pred.clone().into_dyn(), y_train.clone().into_dyn());
        model.backward(grad);
        optimizer_adam.update(model.params_mut());

        if let Some(choice) = validate.choose_mut(&mut rng) {
          let (x_val, y_val) = choice;
          let y_val = one_hot_encode(&y_val, 10);
          let x_val = &*x_val / 255.0;
          let y_pred = model.forward(x_val.clone().into_dyn());
          let loss_value = loss.forward(y_pred.clone(), y_val.clone().into_dyn());
          println!("Validation Loss: {}", loss_value);
        } else {
          println!("バリデーションデータが空です。");
        }
      }
    }
  }

  use csv::Writer;
  use std::collections::HashMap;
  use std::io::{self, Write};

  #[test]
  fn benchmark_optimizers() -> Result<(), Box<dyn std::error::Error>> {
    // MNISTデータの読み込み
    let mnist = load_csv_to_ndarray("data/mnist_train.csv", true).unwrap();
    // オプティマイザの設定
    let optimizers: Vec<(&str, Box<dyn Optimizer>)> = vec![
      ("SGD", Box::new(SGD::new(0.05))),
      ("Momentum", Box::new(Momentum::new(0.05, 0.9))),
      ("RMSProp", Box::new(RMSProp::new(0.005, 0.9))),
      ("Adam", Box::new(Adam::new(0.005, 0.9, 0.999))),
    ];
    // 各オプティマイザの損失を記録するためのハッシュマップ
    let mut loss_records: HashMap<String, Vec<f32>> = HashMap::new();
    // 各オプティマイザでトレーニング
    for (name, mut optimizer) in optimizers {
      println!("Training with {}", name);
      // モデルの初期化（各オプティマイザで同じ構造を使用）
      let he_init = HeInitializer;
      let zero_init = ZeroInitializer;
      let mut model = Sequential::new(vec![
        Box::new(AffineLayer::new(784, 128, &he_init, &zero_init)),
        Box::new(ReLU::new()),
        Box::new(AffineLayer::new(128, 64, &he_init, &zero_init)),
        Box::new(ReLU::new()),
        Box::new(AffineLayer::new(64, 10, &he_init, &zero_init)),
        Box::new(Softmax::new()),
      ]);
      let loss_fn = CrossEntropyLoss::new();
      let mut losses = Vec::new();
      let mut step = 0;
      for _ in 0..10 {
        for (x_train, y_train) in create_batches(&mnist, 0, 2048) {
          let y_train = one_hot_encode(&y_train, 10);
          let x_train = x_train / 255.0;
          let y_pred = model.forward(x_train.clone().into_dyn());
          let loss_value = loss_fn.forward(y_pred.clone(), y_train.clone().into_dyn());
          losses.push(loss_value as f32);
          let grad = loss_fn.backward(y_pred.clone().into_dyn(), y_train.clone().into_dyn());
          model.backward(grad);
          optimizer.update(model.params_mut());
          step += 1;
          print!(
            "\rOptimizer: {}, Step: {}, loss: {}",
            name, step, loss_value
          );
          io::stdout().flush().unwrap();
        }
      }
      loss_records.insert(name.to_string(), losses);
    }
    // CSVファイルへの書き込み
    let mut wtr = Writer::from_path("optimizer_losses.csv")?;
    // ヘッダーの書き込み
    let headers: Vec<&str> = loss_records.keys().map(|k| k.as_str()).collect();
    wtr.write_record(&headers)?;
    // 最大の損失数を取得
    let max_len = loss_records.values().map(|v| v.len()).max().unwrap_or(0);
    // 各ステップの損失を行として書き込み
    for i in 0..max_len {
      let row: Vec<String> = headers
        .iter()
        .map(|&name| {
          loss_records
            .get(name)
            .and_then(|v| v.get(i))
            .map(|val| val.to_string())
            .unwrap_or_else(|| "".to_string())
        })
        .collect();
      wtr.write_record(&row)?;
    }
    wtr.flush()?;
    println!("Losses saved to optimizer_losses.csv");
    Ok(())
  }

  #[test]
  fn test_softmax_and_loss() {
    let mut softmax = Softmax::new();
    let loss = CrossEntropyLoss::new();
    let input = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 5.0, 3.0, 2.0, 5.0, 3.0]).unwrap();
    let output = softmax.forward(input.clone());
    println!("Softmax forward output: {:?}", output);
    let target =
      ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let grad = loss.backward(output.clone(), target.clone());
    let grad = softmax.backward(grad);
    println!("Softmax backward output: {:?}", grad);
  }
}
