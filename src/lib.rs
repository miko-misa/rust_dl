mod clarray;
mod losses;
mod networks;
mod optimizers;
mod params;
mod utils;

#[cfg(test)]
mod tests {
  use crate::clarray::env::{GPUEnv, env};
  use crate::clarray::tensor::{GPUMatrix, GPUTensor, GPUVector, Matrix, Vector};
  use crate::losses::{CrossEntropyLoss, LossFunction};
  use crate::networks::block::Sequential;
  use crate::networks::layer::{AffineLayer, Layer, ReLU, Softmax};
  use crate::optimizers::{Optimizer, SGD};
  use crate::params::initializer::HeInitializer;
  use crate::params::param::LearnableParameter;
  use crate::utils::load_and_prepare_batches;
  use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
  use ocl::core::{DeviceInfo, DeviceInfoResult};
  use ocl::ffi::libc::pause;
  use ocl::{Device, Platform};

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
  fn gpu_vram() -> Result<(), Box<dyn std::error::Error>> {
    // 利用可能な全てのプラットフォームを取得
    let platforms = Platform::list();

    // 各プラットフォームをループ
    for platform in platforms {
      println!("Platform: {}", platform.name()?);

      // プラットフォームに属する全てのデバイスを取得
      let devices = Device::list_all(platform)?;

      // 各デバイスをループ
      for device in devices {
        println!("  Device: {}", device.name()?);

        // デバイスのグローバルメモリ（VRAM）サイズを取得
        match device.info(DeviceInfo::GlobalMemSize)? {
          DeviceInfoResult::GlobalMemSize(vram_bytes) => {
            // バイトをメガバイトに変換
            let vram_mb = vram_bytes / 1024 / 1024;
            println!("    VRAM: {} MB", vram_mb);
          }
          // GlobalMemSizeはU64で返されることが保証されているが、
          // 念のため他のパターンも記述
          _ => {
            println!("    VRAM: Could not determine size.");
          }
        }
      }
      println!("---");
    }
    Ok(())
  }
  #[test]
  fn clarray_test() -> Result<(), Box<dyn std::error::Error>> {
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
      let row = (&row_mat + ((i + 1) as f64))?;
      println!("Row after increment: {:?}", row.to_cpu()?.data);
      row_mat.write(&row)?;
    }
    println!("Matrix A after row-wise increment: {:?}", a.to_cpu()?.data);
    let a = Matrix::from_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.to_gpu()?;
    let b = a.mapv(|x| x * 2.0)?;
    println!("Matrix A after mapv: {:?}", b.to_cpu()?.data);
    let b = a.clip(2.0, f64::MAX)?;
    println!("Matrix A after clipping: {:?}", b.to_cpu()?.data);
    Ok(())
  }

  #[test]
  fn clarray_gemm_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    // 100回行い、平均時間を計測
    let mut total_time = 0.0;
    for _ in 0..100 {
      let a = Matrix::from_vec(
        [1000, 1000],
        (0..1000000).map(|x| x as f64).collect::<Vec<f64>>(),
      )?
      .to_gpu()?;
      let b = Matrix::from_vec(
        [1000, 1000],
        (0..1000000).map(|x| x as f64).collect::<Vec<f64>>(),
      )?
      .to_gpu()?;
      let start_time = std::time::Instant::now();
      let _c = a.dot(&b)?;
      let elapsed_time = start_time.elapsed().as_secs_f64();
      total_time += elapsed_time;
      env().queue.finish().expect("Failed to finish OpenCL queue");
    }
    let average_time = total_time / 100.0;
    println!(
      "Average time for 1000x1000 matrix multiplication: {:.6} seconds",
      average_time
    );
    Ok(())
  }

  #[test]
  fn gpu_deep_learning_test() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 2048;
    let (batches, epoch_step) = load_and_prepare_batches("data/mnist_train.csv", batch_size);
    let mut model = Sequential::new(vec![
      Box::new(AffineLayer::new(784, 128, &HeInitializer)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(128, 64, &HeInitializer)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(64, 10, &HeInitializer)),
      // Box::new(ReLU::new()),
      Box::new(Softmax::new()),
    ]);
    let mut loss_fn = CrossEntropyLoss;
    let mut optimizer = SGD::new(0.1);

    let m = MultiProgress::new();

    for epoch in 0..10 {
      let epoch_pb = m.add(ProgressBar::new(epoch_step as u64));
      epoch_pb.set_style(
        ProgressStyle::default_bar()
          .template(&format!(
            "[Train] Epoch #{}: [{{bar:40.green/black}}] {{pos}}/{{len}} ({{elapsed_precise}} ETA:{{eta}}) {{msg}}",
            epoch + 1
          ))
          .unwrap()
          .progress_chars("##-"),
      );

      for batch in &batches {
        let x = batch.x_flat.clone();
        let y = batch.y_flat.clone();
        let size = y.len() / 10;
        let x = Matrix::from_vec([size, 784], x)?.to_gpu()?;
        let y = Matrix::from_vec([size, 10], y)?.to_gpu()?;
        let output = model.forward(x);
        let loss = loss_fn.forward(output.clone(), y.clone());
        epoch_pb.set_message(format!("Loss: {:.6}", loss));
        let grad = loss_fn.backward(output, y);
        model.backward(grad);
        optimizer.update(model.params_mut());
        epoch_pb.inc(1);
      }
      epoch_pb.finish_with_message(format!("Epoch #{} completed", epoch + 1));
    }
    Ok(())
  }

  #[test]
  fn softmax_and_cross_entropy_test() -> Result<(), Box<dyn std::error::Error>> {
    let x = Matrix::from_vec(
      [3, 4],
      vec![
        1.764052, 0.400157, 0.978738, 2.240893, 1.867558, -0.977278, 0.950088, -0.151357,
        -0.103219, 0.410599, 0.144044, 1.454274,
      ],
    )?
    .to_gpu()?;
    let y = Matrix::from_vec(
      [3, 4],
      vec![
        1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000,
        1.000000, 0.000000, 0.000000,
      ],
    )?
    .to_gpu()?;
    let mut softmax_layer = Softmax::new();
    let output = softmax_layer.forward(x);
    println!("Softmax Output: {:?}", output.to_cpu()?.data);

    let mut loss_fn = CrossEntropyLoss;
    let loss = loss_fn.forward(output.clone(), y.clone());
    println!("Cross Entropy Loss: {:.6}", loss);

    let grad = loss_fn.backward(output, y);
    println!("Gradient: {:?}", grad.to_cpu()?.data);
    let grad = softmax_layer.backward(grad);
    println!("Gradient: {:?}", grad.to_cpu()?.data);

    Ok(())
  }

  #[test]
  fn optimizer_test() -> Result<(), Box<dyn std::error::Error>> {
    let mut param = LearnableParameter {
      id: 1,
      value: Box::new(GPUMatrix::from_vec([1, 3], vec![1.0, 2.0, 3.0], env())?),
      grads: Box::new(GPUMatrix::from_vec([1, 3], vec![0.1, 0.2, 0.3], env())?),
    };
    let mut optimizer = SGD::new(1.0);
    for _ in 0..10 {
      optimizer.update(vec![&mut param]);
      println!(
        "Updated Parameter Value: {:?}",
        param
          .value
          .as_any()
          .downcast_ref::<GPUMatrix<f64>>()
          .unwrap()
          .to_cpu()?
          .data
      );
      param.value = Box::new(GPUMatrix::from_vec([1, 3], vec![1.0, 2.0, 3.0], env())?);
    }

    Ok(())
  }
}
