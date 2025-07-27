use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::{rng, thread_rng};
use std::fs::File;
use std::io::BufReader;

/// バッチ構造体
pub struct Batch {
  pub x_flat: Vec<f64>,
  pub y_flat: Vec<f64>,
}

/// CSVを読み込み、one_hotと正規化、シャッフル、バッチ化を行う関数
pub fn load_and_prepare_batches(path: &str, batch_size: usize) -> (Vec<Batch>, usize) {
  // ファイルオープン
  let file = File::open(path).expect("Cannot open file");
  let mut rdr = ReaderBuilder::new()
    .has_headers(true) // 1行目スキップ
    .from_reader(BufReader::new(file));

  // 全データ格納用
  let mut x_data: Vec<Vec<f64>> = Vec::new();
  let mut y_data: Vec<Vec<f64>> = Vec::new();

  for result in rdr.records() {
    let record = result.expect("Invalid record");
    let mut row = record
      .iter()
      .map(|s| s.parse::<u32>().unwrap())
      .collect::<Vec<u32>>();

    // 1列目: ラベル (0~9) -> one-hot
    let label = row[0] as usize;
    let mut one_hot = vec![0f64; 10];
    one_hot[label] = 1.0;
    y_data.push(one_hot);

    // 2~785列目: 入力 -> f64 + 正規化
    let input = row[1..]
      .iter()
      .map(|&v| v as f64 / 255.0)
      .collect::<Vec<f64>>();
    x_data.push(input);
  }

  // 行ごとに結合してシャッフル
  let mut combined: Vec<(Vec<f64>, Vec<f64>)> = x_data.into_iter().zip(y_data).collect();
  combined.shuffle(&mut rng());

  // 分離
  let (x_data, y_data): (Vec<_>, Vec<_>) = combined.into_iter().unzip();

  // バッチに分割
  let mut batches = Vec::new();
  let total = x_data.len();
  let mut i = 0;
  while i < total {
    let end = (i + batch_size).min(total);
    let x_batch_flat = x_data[i..end].concat(); // flatten
    let y_batch_flat = y_data[i..end].concat(); // flatten
    batches.push(Batch {
      x_flat: x_batch_flat,
      y_flat: y_batch_flat,
    });
    i = end;
  }

  let num_batches = batches.len();
  (batches, num_batches)
}

use std::error::Error;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
struct InnerData {
  rows: Vec<Vec<f64>>,
  labels: Vec<usize>,
}

/// x, y, batch_len を返すイテレータ。Clone 対応。
pub struct BatchIter {
  inner: Arc<InnerData>,
  num_classes: usize,
  batch_size: usize,
  cursor: usize,
}

impl Clone for BatchIter {
  fn clone(&self) -> Self {
    BatchIter {
      inner: Arc::clone(&self.inner),
      num_classes: self.num_classes,
      batch_size: self.batch_size,
      cursor: 0, // クローンは先頭から開始
    }
  }
}

impl Iterator for BatchIter {
  type Item = (Vec<f64>, Vec<f64>, usize);

  fn next(&mut self) -> Option<Self::Item> {
    let total = self.inner.rows.len();
    if self.cursor >= total {
      return None;
    }
    let end = (self.cursor + self.batch_size).min(total);
    let batch_len = end - self.cursor;
    let feature_len = self.inner.rows[0].len();

    let mut x = Vec::with_capacity(batch_len * feature_len);
    let mut y = Vec::with_capacity(batch_len * self.num_classes);

    for i in self.cursor..end {
      x.extend_from_slice(&self.inner.rows[i]);
      let mut one = vec![0.0f64; self.num_classes];
      let cls = self.inner.labels[i];
      one[cls] = 1.0;
      y.extend_from_slice(&one);
    }

    self.cursor = end;
    Some((x, y, batch_len))
  }
}

pub fn load_csv_as_batches<P: AsRef<Path>>(
  path: P,
  batch_size: usize,
  num_classes: usize,
) -> Result<BatchIter, Box<dyn Error>> {
  let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;

  let mut rows = Vec::new();
  let mut labels = Vec::new();
  for result in rdr.records() {
    let record = result?;
    let mut vals: Vec<f64> = record
      .iter()
      .map(|s| s.parse::<f64>().expect("parse error"))
      .collect();
    let label = vals.pop().expect("no label") as usize;
    rows.push(vals);
    labels.push(label);
  }

  // シャッフル
  let mut idx: Vec<_> = (0..rows.len()).collect();
  idx.shuffle(&mut thread_rng());

  let mut s_rows = Vec::with_capacity(rows.len());
  let mut s_labels = Vec::with_capacity(labels.len());
  for i in idx {
    s_rows.push(rows[i].clone());
    s_labels.push(labels[i]);
  }

  let inner = InnerData {
    rows: s_rows,
    labels: s_labels,
  };
  Ok(BatchIter {
    inner: Arc::new(inner),
    num_classes,
    batch_size,
    cursor: 0,
  })
}
