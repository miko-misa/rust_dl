use std::{
  collections::HashMap,
  sync::{Arc, Mutex},
};

use ocl::{Context, Device, Platform, Program, Queue};
use once_cell::sync::Lazy;

pub struct GPUEnv {
  pub context: Context,
  pub device: Device,
  pub platform: Platform,
  pub queue: Queue,
  program: Mutex<HashMap<String, Arc<Program>>>,
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
      program: Mutex::new(HashMap::new()),
    })
  }

  pub fn get_or_compile_program(
    &self,
    kernel_key: &str,
    source_generator: impl Fn() -> String,
  ) -> Result<Arc<Program>, ocl::Error> {
    let mut program_map = self.program.lock().unwrap();
    if let Some(program) = program_map.get(kernel_key) {
      return Ok(program.clone());
    }

    let source = source_generator();
    let program = Program::builder()
      .devices(self.device.clone())
      .src(source)
      .build(&self.context)?;

    let program_arc = Arc::new(program);
    program_map.insert(kernel_key.to_string(), program_arc.clone());
    Ok(program_arc)
  }
}

static GPU_ENV: Lazy<Arc<GPUEnv>> =
  Lazy::new(|| Arc::new(GPUEnv::new().expect("Failed to create GPU environment")));

pub fn env() -> Arc<GPUEnv> {
  GPU_ENV.clone()
}
