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
  use crate::losses::{self, CrossEntropyLoss, CrossEntropyLossWithSoftmax, LossFunction};
  use crate::networks::block::{DimensionConverter, Sequential};
  use crate::networks::layer::{AffineLayer, Conv2D, Flatten, Layer, MaxPooling2D, ReLU, Softmax};
  use crate::optimizers::{Optimizer, SGD};
  use crate::params::initializer::HeInitializer;
  use crate::params::param::LearnableParameter;
  use crate::utils::{load_and_prepare_batches, load_csv_as_batches};
  use core::str;
  use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
  use ocl::core::{DeviceInfo, DeviceInfoResult};
  use ocl::ffi::libc::pause;
  use ocl::{Device, Platform};
  use std::fs::File;
  use std::io::Write;

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
        let exts = device
          .info(ocl::enums::DeviceInfo::Extensions)
          .unwrap()
          .to_string();
        println!("{}", exts);

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
    let vec = GPUVector::from_vec([3], vec![1.0, 2.0, 3.0], env())?;
    println!("Vector: {:?}", vec.to_cpu()?.data);
    let vec_broadcasted = vec.broadcast_matrix(2)?;
    let c = (&vec_broadcasted + &a)?;
    println!("Broadcasted Vector: {:?}", c.to_cpu()?.data);
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
      // Box::new(Softmax::new()),
    ]);
    let mut loss_fn = CrossEntropyLossWithSoftmax;
    let mut optimizer = SGD::new(0.05);

    let mut losses = Vec::new();
    let mut times = Vec::new();

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
        let start_time = std::time::Instant::now();
        let output = model.forward(x);
        let loss = loss_fn.forward(output.clone(), y.clone());
        let grad = loss_fn.backward(output, y);
        model.backward(grad);
        optimizer.update(model.params_mut());
        let elapsed_time = start_time.elapsed().as_millis();
        times.push(elapsed_time);
        epoch_pb.set_message(format!("Loss: {:.6}", loss));
        losses.push(loss);
        epoch_pb.inc(1);
      }
      epoch_pb.finish_with_message(format!("Epoch #{} completed", epoch + 1));
    }
    // save losses to a csv file
    let mut file = std::fs::File::create("losses.csv")?;
    for (loss, time) in losses.iter().zip(times.iter()) {
      // write loss and time to the file
      writeln!(file, "{},{}", loss, time)?;
    }
    Ok(())
  }

  #[test]
  fn gpu_conv_test() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 1;
    // let (batches, epoch_step) = load_and_prepare_batches("data/mnist_train.csv", batch_size);
    let input_shape = [1, 4, 4];
    let fil3x3 = [3, 3];
    let stride = [1, 1];
    let mut conv = Conv2D::new(input_shape, 1, fil3x3, stride, true, &HeInitializer);
    let input = GPUTensor::from_vec(
      [1, input_shape[0], input_shape[1], input_shape[2]],
      vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
      ],
      env(),
    )?;
    let weight = GPUMatrix::from_vec(
      [input_shape[0] * fil3x3[0] * fil3x3[1], 1],
      vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
      env(),
    )?;
    // conv.set_weight(&weight);
    let output = conv.forward(input.clone());
    let mut relu = ReLU::new();
    let output = relu.forward(output);
    println!("Conv Output Shape: {:?}", output.shape);
    println!("Conv Output: {:?}", output.to_cpu()?.data);
    let mut pool = MaxPooling2D::new(
      [output.shape[1], output.shape[2], output.shape[3]],
      [2, 2],
      [2, 2],
    );
    let output = pool.forward(output);
    println!("Pooling Output Shape: {:?}", output.shape);
    println!("Pooling Output: {:?}", output.to_cpu()?.data);
    Ok(())
  }

  struct CNN_Block {
    input_shape: [usize; 3],
    output_channels: usize,
    conv: Conv2D,
    relu: ReLU<[usize; 4]>,
    pool: MaxPooling2D,
  }

  impl CNN_Block {
    pub fn new(input_shape: [usize; 3], output_channels: usize) -> Self {
      let conv = Conv2D::new(
        input_shape,
        output_channels,
        [3, 3],
        [1, 1],
        true,
        &HeInitializer,
      );
      let relu = ReLU::new();
      let pool = MaxPooling2D::new(
        [output_channels, input_shape[1], input_shape[2]],
        [2, 2],
        [2, 2],
      );
      CNN_Block {
        input_shape,
        output_channels,
        conv,
        relu,
        pool,
      }
    }
  }

  impl Layer<[usize; 4], [usize; 4]> for CNN_Block {
    fn forward(&mut self, input: GPUTensor<f64, [usize; 4]>) -> GPUTensor<f64, [usize; 4]> {
      let conv_output = self.conv.forward(input);
      let relu_output = self.relu.forward(conv_output);
      self.pool.forward(relu_output)
    }

    fn backward(&mut self, grad: GPUTensor<f64, [usize; 4]>) -> GPUTensor<f64, [usize; 4]> {
      let pool_grad = self.pool.backward(grad);
      let relu_grad = self.relu.backward(pool_grad);
      self.conv.backward(relu_grad)
    }

    fn params_mut(&mut self) -> Vec<&mut LearnableParameter<f64>> {
      let mut params = self.conv.params_mut();
      params.extend(self.relu.params_mut());
      params.extend(self.pool.params_mut());
      params
    }

    fn set_training(&mut self, training: bool) {
      self.conv.set_training(training);
      self.relu.set_training(training);
      self.pool.set_training(training);
    }
  }

  #[test]
  fn cnn_block_test() -> Result<(), Box<dyn std::error::Error>> {
    let input_shape = [3, 224, 224];
    let output_channels = 64;
    let mut cnn_block = CNN_Block::new(input_shape, output_channels);
    let input = GPUTensor::from_vec(
      [1, input_shape[0], input_shape[1], input_shape[2]],
      (0..(3 * 224 * 224))
        .map(|x| ((x as f64 % 256.0) / 256.0))
        .collect::<Vec<f64>>(),
      env(),
    )?;
    let output = cnn_block.forward(input);
    println!("Output shape: {:?}", output.shape);
    println!("Output data: {:?}", output.to_cpu()?.data);
    Ok(())
  }

  #[test]
  fn cnn_test() -> Result<(), Box<dyn std::error::Error>> {
    let conv_model = Sequential::new(vec![
      Box::new(CNN_Block::new([3, 32, 32], 64)),
      Box::new(CNN_Block::new([64, 16, 16], 128)),
      Box::new(CNN_Block::new([128, 8, 8], 256)),
    ]);
    let dense_model = Sequential::new(vec![
      Box::new(AffineLayer::new(256 * 4 * 4, 4096, &HeInitializer)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(4096, 4096, &HeInitializer)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(4096, 10, &HeInitializer)),
    ]);
    let mut model = DimensionConverter::new(
      Box::new(conv_model),
      Box::new(Flatten::new()),
      Box::new(dense_model),
    );
    let mut loss_fn = CrossEntropyLossWithSoftmax;
    let mut optimizer = SGD::new(0.01);
    println!("prepareing input tensor");
    let input = GPUTensor::from_vec(
      [2, 3, 32, 32],
      (0..(2 * 3 * 32 * 32))
        .map(|x| ((x as f64 % 256.0) / 256.0))
        .collect::<Vec<f64>>(),
      env(),
    )?;
    let y_ture = GPUTensor::from_vec(
      [2, 10],
      vec![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
      ],
      env(),
    )?;
    println!("start");
    for _ in 0..100 {
      let output = model.forward(input.clone());
      let loss = loss_fn.forward(output.clone(), y_ture.clone());
      println!("Loss: {:.6}", loss);
      let grad = loss_fn.backward(output, y_ture.clone());
      model.backward(grad);
      //println!("Backward pass completed");
      optimizer.update(model.params_mut());
      //println!("Updated model parameters");
    }
    Ok(())
  }

  #[test]
  fn cnn_learn() -> Result<(), Box<dyn std::error::Error>> {
    let conv_model = Sequential::new(vec![
      Box::new(CNN_Block::new([3, 32, 32], 64)),
      Box::new(CNN_Block::new([64, 16, 16], 128)),
      Box::new(CNN_Block::new([128, 8, 8], 256)),
    ]);
    let dense_model = Sequential::new(vec![
      Box::new(AffineLayer::new(256 * 4 * 4, 4096, &HeInitializer)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(4096, 4096, &HeInitializer)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(4096, 100, &HeInitializer)),
    ]);
    let mut model = DimensionConverter::new(
      Box::new(conv_model),
      Box::new(Flatten::new()),
      Box::new(dense_model),
    );
    let mut loss_fn = CrossEntropyLossWithSoftmax;
    let mut optimizer = SGD::new(0.05);
    let mut losses = Vec::new();
    let mut times = Vec::new();
    for _ in 0..15 {
      println!("prepareing input tensor");
      let data = load_csv_as_batches("data/cifar100_train.csv", 256, 100)?;
      println!("start");
      for (x, y, n) in data.clone() {
        let start_time = std::time::Instant::now();
        let input = GPUTensor::from_vec([n, 3, 32, 32], x, env())?;
        let y_ture = GPUTensor::from_vec([n, 100], y, env())?;
        let output = model.forward(input.clone());
        let loss = loss_fn.forward(output.clone(), y_ture.clone());
        let grad = loss_fn.backward(output, y_ture.clone());
        model.backward(grad);
        //println!("Backward pass completed");
        optimizer.update(model.params_mut());
        //println!("Updated model parameters");
        let elapsed_time = start_time.elapsed().as_secs_f64();
        println!("Loss: {:.6}, Time: {:.2}s", loss, elapsed_time);
        losses.push(loss);
        times.push(elapsed_time);
      }
    }
    // save to csv file
    let mut file = std::fs::File::create("cnn_losses.csv")?;
    for (loss, time) in losses.iter().zip(times.iter()) {
      writeln!(file, "{},{}", loss, time)?;
    }
    println!("Training completed and losses saved to cnn_losses.csv");
    Ok(())
  }

  #[test]
  fn softmax_and_cross_entropy_test() -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..1 {
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
    }
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

  #[test]
  fn im2col_test() -> Result<(), Box<dyn std::error::Error>> {
    let tensor = GPUTensor::from_vec(
      [1, 1, 4, 4],
      vec![
        1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000,
        10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 16.000000,
      ],
      env(),
    )?;
    println!("tensor strides: {:?}", tensor.strides);
    let filter_size = [2, 2];
    let stride = [1, 1];
    let col = tensor.im2col(
      [tensor.shape[1], filter_size[0], filter_size[1]],
      [1, stride[0], stride[1]],
    )?;
    // println!("Col: {:?}", col.to_cpu()?.data);
    // fileにカンマ区切りで出力
    let mut file = File::create("col_output.csv")?;
    for row in col.to_cpu()?.data.chunks(col.shape[1]) {
      writeln!(
        file,
        "{}",
        row
          .iter()
          .map(|x| x.to_string())
          .collect::<Vec<_>>()
          .join(",")
      )?;
    }
    Ok(())
  }

  #[test]
  fn col2im_test() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = GPUMatrix::from_vec(
      [42, 12],
      vec![
        0.651000, 0.034000, 0.551000, 0.804000, 0.649000, 0.232000, 0.378000, 0.337000, 0.081000,
        0.458000, 0.107000, 0.494000, 0.034000, 0.551000, 0.761000, 0.649000, 0.232000, 0.553000,
        0.337000, 0.081000, 0.669000, 0.107000, 0.494000, 0.310000, 0.551000, 0.761000, 0.697000,
        0.232000, 0.553000, 0.374000, 0.081000, 0.669000, 0.696000, 0.494000, 0.310000, 0.546000,
        0.761000, 0.697000, 0.087000, 0.553000, 0.374000, 0.303000, 0.669000, 0.696000, 0.774000,
        0.310000, 0.546000, 0.969000, 0.697000, 0.087000, 0.026000, 0.374000, 0.303000, 0.335000,
        0.696000, 0.774000, 0.371000, 0.546000, 0.969000, 0.127000, 0.087000, 0.026000, 0.845000,
        0.303000, 0.335000, 0.697000, 0.774000, 0.371000, 0.443000, 0.969000, 0.127000, 0.460000,
        0.804000, 0.649000, 0.232000, 0.457000, 0.171000, 0.131000, 0.458000, 0.107000, 0.494000,
        0.790000, 0.119000, 0.557000, 0.649000, 0.232000, 0.553000, 0.171000, 0.131000, 0.685000,
        0.107000, 0.494000, 0.310000, 0.119000, 0.557000, 0.598000, 0.232000, 0.553000, 0.374000,
        0.131000, 0.685000, 0.539000, 0.494000, 0.310000, 0.546000, 0.557000, 0.598000, 0.254000,
        0.553000, 0.374000, 0.303000, 0.685000, 0.539000, 0.316000, 0.310000, 0.546000, 0.969000,
        0.598000, 0.254000, 0.013000, 0.374000, 0.303000, 0.335000, 0.539000, 0.316000, 0.096000,
        0.546000, 0.969000, 0.127000, 0.254000, 0.013000, 0.176000, 0.303000, 0.335000, 0.697000,
        0.316000, 0.096000, 0.684000, 0.969000, 0.127000, 0.460000, 0.013000, 0.176000, 0.584000,
        0.457000, 0.171000, 0.131000, 0.552000, 0.597000, 0.602000, 0.790000, 0.119000, 0.557000,
        0.370000, 0.760000, 0.948000, 0.171000, 0.131000, 0.685000, 0.597000, 0.602000, 0.034000,
        0.119000, 0.557000, 0.598000, 0.760000, 0.948000, 0.071000, 0.131000, 0.685000, 0.539000,
        0.602000, 0.034000, 0.593000, 0.557000, 0.598000, 0.254000, 0.948000, 0.071000, 0.833000,
        0.685000, 0.539000, 0.316000, 0.034000, 0.593000, 0.619000, 0.598000, 0.254000, 0.013000,
        0.071000, 0.833000, 0.962000, 0.539000, 0.316000, 0.096000, 0.593000, 0.619000, 0.645000,
        0.254000, 0.013000, 0.176000, 0.833000, 0.962000, 0.275000, 0.316000, 0.096000, 0.684000,
        0.619000, 0.645000, 0.117000, 0.013000, 0.176000, 0.584000, 0.962000, 0.275000, 0.958000,
        0.552000, 0.597000, 0.602000, 0.629000, 0.555000, 0.625000, 0.370000, 0.760000, 0.948000,
        0.463000, 0.153000, 0.730000, 0.597000, 0.602000, 0.034000, 0.555000, 0.625000, 0.247000,
        0.760000, 0.948000, 0.071000, 0.153000, 0.730000, 0.234000, 0.602000, 0.034000, 0.593000,
        0.625000, 0.247000, 0.571000, 0.948000, 0.071000, 0.833000, 0.730000, 0.234000, 0.267000,
        0.034000, 0.593000, 0.619000, 0.247000, 0.571000, 0.508000, 0.071000, 0.833000, 0.962000,
        0.234000, 0.267000, 0.417000, 0.593000, 0.619000, 0.645000, 0.571000, 0.508000, 0.243000,
        0.833000, 0.962000, 0.275000, 0.267000, 0.417000, 0.065000, 0.619000, 0.645000, 0.117000,
        0.508000, 0.243000, 0.677000, 0.962000, 0.275000, 0.958000, 0.417000, 0.065000, 0.545000,
        0.629000, 0.555000, 0.625000, 0.551000, 0.195000, 0.588000, 0.463000, 0.153000, 0.730000,
        0.560000, 0.982000, 0.629000, 0.555000, 0.625000, 0.247000, 0.195000, 0.588000, 0.911000,
        0.153000, 0.730000, 0.234000, 0.982000, 0.629000, 0.198000, 0.625000, 0.247000, 0.571000,
        0.588000, 0.911000, 0.161000, 0.730000, 0.234000, 0.267000, 0.629000, 0.198000, 0.497000,
        0.247000, 0.571000, 0.508000, 0.911000, 0.161000, 0.910000, 0.234000, 0.267000, 0.417000,
        0.198000, 0.497000, 0.987000, 0.571000, 0.508000, 0.243000, 0.161000, 0.910000, 0.881000,
        0.267000, 0.417000, 0.065000, 0.497000, 0.987000, 0.310000, 0.508000, 0.243000, 0.677000,
        0.910000, 0.881000, 0.587000, 0.417000, 0.065000, 0.545000, 0.987000, 0.310000, 0.373000,
        0.551000, 0.195000, 0.588000, 0.340000, 0.135000, 0.514000, 0.560000, 0.982000, 0.629000,
        0.842000, 0.282000, 0.572000, 0.195000, 0.588000, 0.911000, 0.135000, 0.514000, 0.125000,
        0.982000, 0.629000, 0.198000, 0.282000, 0.572000, 0.057000, 0.588000, 0.911000, 0.161000,
        0.514000, 0.125000, 0.963000, 0.629000, 0.198000, 0.497000, 0.572000, 0.057000, 0.839000,
        0.911000, 0.161000, 0.910000, 0.125000, 0.963000, 0.964000, 0.198000, 0.497000, 0.987000,
        0.057000, 0.839000, 0.072000, 0.161000, 0.910000, 0.881000, 0.963000, 0.964000, 0.027000,
        0.497000, 0.987000, 0.310000, 0.839000, 0.072000, 0.258000, 0.910000, 0.881000, 0.587000,
        0.964000, 0.027000, 0.979000, 0.987000, 0.310000, 0.373000, 0.072000, 0.258000, 0.707000,
        0.340000, 0.135000, 0.514000, 0.277000, 0.509000, 0.560000, 0.842000, 0.282000, 0.572000,
        0.407000, 0.022000, 0.719000, 0.135000, 0.514000, 0.125000, 0.509000, 0.560000, 0.369000,
        0.282000, 0.572000, 0.057000, 0.022000, 0.719000, 0.299000, 0.514000, 0.125000, 0.963000,
        0.560000, 0.369000, 0.609000, 0.572000, 0.057000, 0.839000, 0.719000, 0.299000, 0.208000,
        0.125000, 0.963000, 0.964000, 0.369000, 0.609000, 0.929000, 0.057000, 0.839000, 0.072000,
        0.299000, 0.208000, 0.274000, 0.963000, 0.964000, 0.027000, 0.609000, 0.929000, 0.646000,
        0.839000, 0.072000, 0.258000, 0.208000, 0.274000, 0.842000, 0.964000, 0.027000, 0.979000,
        0.929000, 0.646000, 0.436000, 0.072000, 0.258000, 0.707000, 0.274000, 0.842000, 0.661000,
      ],
      env(),
    )?;
    let img_size = [2, 8, 8];
    let filter_size = [2, 3];
    let stride = [1, 1];
    let tensor = matrix.col2im(
      [img_size[0], filter_size[0], filter_size[1]],
      [1, stride[0], stride[1]],
      [1, img_size[0], img_size[1], img_size[2]],
    )?;
    println!("Tensor shape: {:?}", tensor.shape);
    println!("Tensor: {:?}", tensor.to_cpu()?.data);
    Ok(())
  }

  #[test]
  fn max_pooling_test() -> Result<(), Box<dyn std::error::Error>> {
    let tensor = GPUTensor::from_vec(
      [1, 1, 4, 4],
      vec![
        1.0, 4.0, 3.0, 2.0, //
        5.0, 6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0, 12.0, //
        13.0, 14.0, 15.0, 16.0,
      ],
      env(),
    )?;
    let filter_size = [2, 2];
    let stride = [2, 2];
    let mut pool_layer = MaxPooling2D::new(
      [tensor.shape[1], tensor.shape[2], tensor.shape[3]],
      filter_size,
      stride,
    );
    let pooled = pool_layer.forward(tensor);
    let grad = GPUTensor::from_vec([1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0], env())?;
    let grad = pool_layer.backward(grad);
    println!("Pooled Tensor: {:?}", pooled.to_cpu()?.data);
    println!("Gradient after MaxPooling: {:?}", grad.to_cpu()?.data);
    Ok(())
  }
}
