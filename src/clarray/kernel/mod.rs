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
