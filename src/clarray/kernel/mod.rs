pub mod matrix;
pub mod vector;

pub fn clang_type_name(type_name: &str) -> String {
  match type_name {
    "f32" => "float",
    "f64" => "double",
    _ => panic!("Unsupported type: {}", type_name),
  }
  .to_string()
}

pub fn op_to_suffix(op: &str) -> String {
  match op {
    "+" => "add",
    "-" => "sub",
    "*" => "mul",
    "/" => "div",
    "%" => "mod",
    _ => panic!("Unsupported operation: {}", op),
  }
  .to_string()
}

pub fn get_all_kernels_as_string(type_suffix: &str) -> String {
  let type_name = clang_type_name(type_suffix);
  let mut all_kernels = String::new();
  let ops = ["+", "-", "*", "/"];

  // --- vector.rsのカーネル ---
  all_kernels.push_str(&format!(
    "\n// Kernels from vector.rs (type: {})\n",
    type_name
  ));

  // 型のみ指定
  all_kernels.push_str(&vector::broadcast_matrix_source(&type_name, &type_suffix));
  all_kernels.push_str(&vector::sum_source(&type_name, &type_suffix));
  all_kernels.push_str(&vector::write_source(&type_name, &type_suffix));

  // 四則演算
  for &op in &ops {
    all_kernels.push_str(&vector::elementwise_op_source(op, &type_name, &type_suffix));
    all_kernels.push_str(&vector::elementwise_op_scalar_r_source(
      op,
      &type_name,
      &type_suffix,
    ));
    all_kernels.push_str(&vector::elementwise_op_scalar_l_source(
      op,
      &type_name,
      &type_suffix,
    ));
  }

  // --- matrix.rsのカーネル ---
  all_kernels.push_str(&format!(
    "\n// Kernels from matrix.rs (type: {})\n",
    type_name
  ));

  // 型のみ指定
  all_kernels.push_str(&matrix::repeak_source(&type_name, type_suffix));
  all_kernels.push_str(&matrix::dot_source(&type_name, type_suffix));
  all_kernels.push_str(&matrix::write_source(&type_name, type_suffix));
  all_kernels.push_str(&matrix::clip_source(&type_name, type_suffix));
  all_kernels.push_str(&matrix::row_sum_source(&type_name, type_suffix));
  all_kernels.push_str(&matrix::diag_source(&type_name, type_suffix));

  // 四則演算
  for &op in &ops {
    all_kernels.push_str(&matrix::elementwise_op_source(op, &type_name, type_suffix));
    all_kernels.push_str(&matrix::elementwise_op_scalar_r_source(
      op,
      &type_name,
      type_suffix,
    ));
    all_kernels.push_str(&matrix::elementwise_op_scalar_l_source(
      op,
      &type_name,
      type_suffix,
    ));
  }

  all_kernels
}
