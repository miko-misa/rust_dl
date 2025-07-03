mod accuracies;
mod clarray;
mod losses;
mod models;
mod networks;
mod optimizers;
mod params;
mod utils;

#[cfg(test)]
mod tests {
  use crate::accuracies::{Accuracy, OnehotArgmaxAccuracy};
  use crate::clarray::env::{GPUEnv, env};
  use crate::clarray::tensor::Matrix;
  use crate::losses::{CrossEntropyLoss, LossFunction};
  use crate::models::BaseModel;
  use crate::networks::layer::{AffineLayer, BatchNorm, Dropout, Layer, ReLU, Softmax};
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
  use ndarray::{Array2, ArrayD, IxDyn};
  use ocl::{Device, Platform};
  use utils::load_csv::load_csv_to_ndarray;

  #[test]
  fn list_devices() -> () {
    // 利用可能なプラットフォームを取得
    let platforms = Platform::list();
    for platform in platforms {
      println!("Platform: {}", platform.name().unwrap());

      // プラットフォームに関連するデバイスを取得
      let devices = Device::list_all(&platform).unwrap();
      for device in devices {
        // デバイスのタイプを取得
        let device_type = device.info(ocl::enums::DeviceInfo::Type).unwrap();
        let device_type_str = match device_type.to_string().as_str() {
          "CPU" => "CPU",
          "GPU" => "GPU",
          "Accelerator" => "Accelerator",
          "Custom" => "Custom",
          "All" => "All",
          _ => "Unknown",
        };
        println!("\tDevice: {} ({})", device.name().unwrap(), device_type_str);
      }
    }
  }
  #[test]
  fn gpu_gemm() -> Result<(), Box<dyn std::error::Error>> {
    let a = Matrix::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.to_gpu()?;
    println!("Matrix A: {:?}", a.to_cpu()?.data);

    let b = (&a + 1.0)?;
    println!("Matrix A + 1: {:?}", b.to_cpu()?.data);

    let b = (&a - 12.0)?;
    println!("Matrix A - 12: {:?}", b.to_cpu()?.data);

    let b = (5.0f64 * &a)?;
    println!("5 * Matrix A: {:?}", b.to_cpu()?.data);

    let b = (1.0f64 / &a)?;
    println!("1 / Matrix A: {:?}", b.to_cpu()?.data);

    let c = Matrix::from_vec([3, 2], vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0])?.to_gpu()?;
    println!("Matrix C: {:?}", c.to_cpu()?.data);
    println!("Hadmard product A * C: {:?}", (&a * &c)?.to_cpu()?.data);

    for (i, row_mat) in a.row_iter().enumerate() {
      println!("Row: {:?}", row_mat.to_cpu()?.data);
      row_mat.write(&(&row_mat + (i as f64))?)?;
    }
    println!("Matrix A after row-wise increment: {:?}", a.to_cpu()?.data);
    Ok(())
  }

  #[test]
  fn it_works() -> Result<(), Box<dyn std::error::Error>> {
    let mnist = load_csv_to_ndarray("data/mnist_train.csv", true).unwrap();
    let mnist_val = load_csv_to_ndarray("data/mnist_test.csv", true).unwrap();
    let he_init = HeInitializer;
    let zero_init = ZeroInitializer;
    let model_layer = Sequential::new(vec![
      Box::new(AffineLayer::new(784, 128, &he_init, &zero_init, None)),
      Box::new(BatchNorm::new(128, 0.9, &he_init, &zero_init)),
      Box::new(ReLU::new()),
      Box::new(Dropout::new(0.2)),
      Box::new(AffineLayer::new(128, 64, &he_init, &zero_init, None)),
      Box::new(BatchNorm::new(64, 0.9, &he_init, &zero_init)),
      Box::new(ReLU::new()),
      Box::new(Dropout::new(0.2)),
      Box::new(AffineLayer::new(64, 10, &he_init, &zero_init, None)),
      Box::new(BatchNorm::new(10, 0.9, &he_init, &zero_init)),
      Box::new(Softmax::new()),
    ]);
    let mut model = BaseModel::new(
      Box::new(model_layer),
      Box::new(CrossEntropyLoss::new()),
      Box::new(Adam::new(0.01, 0.9, 0.999)),
      Box::new(OnehotArgmaxAccuracy::new()),
    );
    let mut train_data = create_batches(&mnist, 0, 10000);
    for (x_train, y_train) in train_data.iter_mut() {
      *y_train = one_hot_encode(&y_train, 10);
      *x_train = &*x_train / 255.0;
    }
    let mut val_data = create_batches(&mnist_val, 0, 4096);
    for (x_test, y_test) in val_data.iter_mut() {
      *y_test = one_hot_encode(&y_test, 10);
      *x_test = &*x_test / 255.0;
    }
    let train_result = model.train_step(
      10,
      &train_data
        .into_iter()
        .map(|(x, y)| (x.into_dyn(), y.into_dyn()))
        .collect(),
      &val_data
        .into_iter()
        .map(|(x, y)| (x.into_dyn(), y.into_dyn()))
        .collect(),
    );

    let mut wtr = Writer::from_path("data/result/output.csv")?;
    wtr.write_record(&["train loss", "val loss", "train acc", "val acc"])?;
    for (train_loss, val_loss, train_acc, val_acc) in train_result {
      wtr.write_record(&[
        train_loss.to_string(),
        val_loss.to_string(),
        train_acc.to_string(),
        val_acc.to_string(),
      ])?;
    }
    wtr.flush()?;
    Ok(())
  }

  use csv::{Error, Writer};
  use std::any::Any;
  use std::collections::HashMap;
  use std::env;
  use std::io::{self, Write};
  use std::sync::Arc;

  #[test]
  fn benchmark_optimizers() -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking optimizers on MNIST dataset...");
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
        Box::new(AffineLayer::new(784, 128, &he_init, &zero_init, None)),
        Box::new(ReLU::new()),
        Box::new(AffineLayer::new(128, 64, &he_init, &zero_init, None)),
        Box::new(ReLU::new()),
        Box::new(AffineLayer::new(64, 10, &he_init, &zero_init, None)),
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

  #[test]
  fn accuracy_test() {
    let pred = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.1, 0.9, 0.0, 0.8, 0.1, 0.1]).unwrap();
    let target =
      ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let accuracy = OnehotArgmaxAccuracy::new();
    let acc = accuracy.accuracy(&pred, &target);
    println!("Accuracy: {:?}", acc);
  }
}
