mod losses;
mod networks;
mod optimizers;
mod utils;

#[cfg(test)]
mod tests {
  use crate::losses::{CrossEntropyLoss, LossFunction};
  use crate::networks::layer::{AffineLayer, Layer, ReLU, Softmax};
  use crate::optimizers::SGD;
  use crate::{
    networks::block::Sequential,
    optimizers::Optimizer,
    utils::{
      self,
      batch::{create_batches, one_hot_encode},
    },
  };
  use ndarray::{ArrayD, Axis, IxDyn};
  use utils::load_csv::load_csv_to_ndarray;

  #[test]
  fn it_works() {
    let mnist = load_csv_to_ndarray("data/mnist_train.csv", true).unwrap();
    let mut model = Sequential::new(vec![
      Box::new(AffineLayer::new(784, 128)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(128, 64)),
      Box::new(ReLU::new()),
      Box::new(AffineLayer::new(64, 10)),
      Box::new(Softmax::new()),
    ]);
    let mut optimizer = SGD::new(0.05);
    let loss = CrossEntropyLoss::new();
    for _ in 0..10 {
      for (x_train, y_train) in create_batches(&mnist, 0, 2048) {
        let y_train = one_hot_encode(&y_train, 10);
        let x_train = x_train / 255.0;
        let y_pred = model.forward(x_train.clone().into_dyn());
        let loss_value = loss.forward(y_pred.clone(), y_train.clone().into_dyn());
        println!("Loss: {}", loss_value);
        let grad = loss.backward(y_pred.clone().into_dyn(), y_train.clone().into_dyn());
        model.backward(grad);
        optimizer.update(model.params_mut());
      }
    }
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
