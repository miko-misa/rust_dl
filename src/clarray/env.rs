use std::{
  collections::HashMap,
  fs::File,
  io::Write,
  sync::{Arc, RwLock},
};

use ocl::{Context, Device, Platform, Program, Queue};
use once_cell::sync::Lazy;

use crate::clarray::kernel::get_all_kernels_as_string;

pub struct GPUEnv {
  pub context: Context,
  pub device: Device,
  pub platform: Platform,
  pub queue: Queue,
  program: RwLock<HashMap<String, Arc<Program>>>,
}

impl GPUEnv {
  pub fn new() -> Result<Self, ocl::Error> {
    let platform = Platform::default();
    let device = Device::first(&platform)?;
    let context = Context::builder()
      .platform(platform)
      .devices(device)
      .build()?;
    let queue = Queue::new(&context, device.clone(), None)?;

    Ok(GPUEnv {
      context,
      device,
      platform,
      queue,
      program: RwLock::new(HashMap::new()),
    })
  }

  pub fn get_or_compile_program(&self, type_suffix: &str) -> Result<Arc<Program>, ocl::Error> {
    let mut programs = self.program.write().unwrap();
    if let Some(program) = programs.get(type_suffix) {
      return Ok(Arc::clone(program));
    }

    let source = get_all_kernels_as_string(type_suffix);
    write_string_to_file(&format!("kernels_{}.cl", type_suffix), &source)
      .expect("Failed to write OpenCL source to file");
    let program = Program::builder()
      .src(source)
      .devices(self.device.clone())
      .build(&self.context)?;

    let program_arc = Arc::new(program);
    programs.insert(type_suffix.to_string(), Arc::clone(&program_arc));

    Ok(program_arc)
  }
}

fn write_string_to_file(filename: &str, content: &str) -> std::io::Result<()> {
  let mut file = File::create(filename)?; // ファイルを作成（または上書き）
  file.write_all(content.as_bytes())?; // 文字列をバイト列として書き込む
  Ok(())
}

static GPU_ENV: Lazy<Arc<GPUEnv>> =
  Lazy::new(|| Arc::new(GPUEnv::new().expect("Failed to create GPU environment")));

pub fn env() -> Arc<GPUEnv> {
  GPU_ENV.clone()
}
