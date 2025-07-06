mod clarray;
mod networks;
mod optimizers;
mod params;

#[cfg(test)]
mod tests {
  use crate::clarray::env::{GPUEnv, env};
  use crate::clarray::tensor::Matrix;
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
    }
    let average_time = total_time / 100.0;
    println!(
      "Average time for 1000x1000 matrix multiplication: {:.6} seconds",
      average_time
    );
    Ok(())
  }
}
