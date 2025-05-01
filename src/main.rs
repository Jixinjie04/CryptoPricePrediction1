use std::error::Error;
use std::fs::File;
use std::io::{Read};
use std::path::Path;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use csv;
use plotters::prelude::*;
use chrono::{DateTime, Utc, TimeZone};
extern crate plotters;
use plotters::prelude::*;
use ndarray::Array1;
use plotters::evcxr::SVGWrapper;
use plotters::style::colors::full_palette;
use ndarray::{Array2, ArrayBase, OwnedRepr, Dim};




#[derive(Debug, Deserialize, Serialize)]
struct DirtyKline {
    #[serde(rename = "Open Time")]
    open_time: u64,
    #[serde(rename = "Open")]
    open: String,
    #[serde(rename = "High")]
    high: String,
    #[serde(rename = "Low")]
    low: String,
    #[serde(rename = "Close")]
    close: String,
    #[serde(rename = "Volume")]
    volume: String,
    #[serde(rename = "Close Time")]
    close_time: u64,
    #[serde(rename = "Quote Asset Volume")]
    quote_asset_volume: String,
    #[serde(rename = "Number of Trades")]
    number_of_trades: u64,
    #[serde(rename = "Taker Buy Base Asset Volume")]
    taker_buy_base_asset_volume: String,
    #[serde(rename = "Taker Buy Quote Asset Volume")]
    taker_buy_quote_asset_volume: String,
    #[serde(rename = "Ignore")]
    ignore: String,
}

#[derive(Debug, Default)]
struct CleanKline {
    open_time: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    close_time: DateTime<Utc>,
    quote_asset_volume: f64,
    number_of_trades: u64,
    taker_buy_base_asset_volume: f64,
    taker_buy_quote_asset_volume: f64,
    ignore: f64,
}



fn CleanRecord(r:DirtyKline) -> CleanKline {
    let mut c = CleanKline::default();
    c.open_time = Utc.timestamp(r.open_time as i64, 0);
    c.open = r.open.parse::<f64>().unwrap();
    c.high = r.high.parse::<f64>().unwrap();
    c.low = r.low.parse::<f64>().unwrap();
    c.close = r.close.parse::<f64>().unwrap();
    c.volume = r.volume.parse::<f64>().unwrap();
    c.close_time = Utc.timestamp(r.close_time as i64, 0);
    c.quote_asset_volume = r.quote_asset_volume.parse::<f64>().unwrap();
    c.number_of_trades = r.number_of_trades;
    c.taker_buy_base_asset_volume = r.taker_buy_base_asset_volume.parse::<f64>().unwrap();
    c.taker_buy_quote_asset_volume = r.taker_buy_quote_asset_volume.parse::<f64>().unwrap();
    c.ignore = r.ignore.parse::<f64>().unwrap();
    c
}

fn processed_csv_file(path: &str) -> Array2<f32> {
    let mut file =  File::open(path).expect("Failed to open CSV file");
    let mut content = String::new();
    file.read_to_string(&mut content).expect("Failed to read CSV file");
    let first_line = content.lines().next().unwrap();
    let headers: Vec<&str> = first_line.split(',').collect();
    let mut v: Vec<DirtyKline> = Vec::new();

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path("klines.csv")
        .expect("Failed to create CSV reader");
    for result in rdr.deserialize::<DirtyKline> () {
        match result {
            Ok(result) => {
                v.push(result);
            },
            Err(err) => {
                println!("Error reading record: {}", err);
                continue;
            }
        }
    }
    let mut clean_klines: Array2<CleanKline> = Vec::new();
    for record in v{
        let clean_record: CleanKline = CleanRecord(record);
        clean_klines.push(clean_record);
    }
    clean_klines
}


struct NeuralNetwork {
    inputs_size: usize,
    layer1_size: usize,
    layer2_size:usize,
    layer3_size:usize,
    layer4_size:usize,
    outputs_size: usize,
    learning_rate: f32,
    weights_input_to_layer1: Array2<f32>,
    weights_layer1_to_layer2: Array2<f32>,
    weights_layer2_to_layer3: Array2<f32>,
    weights_layer3_to_layer4: Array2<f32>,
    weights_layer4_to_output: Array2<f32>,
}

impl NeuralNetwork {
    fn new(inputs_size: usize, layer1_size: usize, layer2_size: usize, layer3_size: usize, layer4_size: usize, outputs_layer: usize, learning_rate: f32) -> Self {
        let weights_input_to_layer1 = Array2::random((inputs_size, layer1_size), Uniform::new(-0.5, 0.5));
        let weights_layer1_to_layer2 = Array2::random((layer1_size, layer2_size), Uniform::new(-0.5,0.5));
        let weights_layer2_to_layer3 = Array2::random((layer2_size, layer3_size), Uniform::new(-0.5,0.5));
        let weights_layer3_to_layer4 = Array2::random((layer3_size, layer4_size), Uniform::new(-0.5, 0.5));
        let weights_layer4_to_output = Array2::random((layer4_size, outputs_layer), Uniform::new(-0.5, 0.5));
        NeuralNetwork {
            inputs_size: inputs_size,
            layer1_size: layer1_size,
            layer2_size: layer2_size,
            layer3_size: layer3_size,
            layer4_size: layer4_size,
            outputs_size: outputs_layer,
            learning_rate: learning_rate,
            weights_input_to_layer1,
            weights_layer1_to_layer2,
            weights_layer2_to_layer3,
            weights_layer3_to_layer4,
            weights_layer4_to_output
        }

    }

    fn sigmoid(input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| 1.0/(1.0 + (-x).exp()))
    }

    fn sigmoid_derivative(input: &Array2<f32>) -> Aray2<f32> {
        input.mapv(|x| x* (1.0-x))
    }

    fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let hidden1 = input.dot(&self.weights_input_to_layer1);
        let hidden1_out = NeuralNetwork::sigmoid(&hidden1);
        let hidden2 = hidden1_out.dot(&self.weights_layer1_to_layer2);
        let hidden2_out = NeuralNetwork::sigmoid(&hidden2);
        let hidden3 = hidden2_out.dot(&self.weights_layer2_to_layer3);
        let hidden3_out = NeuralNetwork::sigmoid(&hidden3);
        let hidden4 = hidden3_out.dot(&self.weights_layer3_to_layer4);
        let hidden4_out = NeuralNetwork::sigmoid(&hidden4);
        let output = hidden4_out.dot(&self.weights_layer4_to_output);
        let output_out = NeuralNetwork::sigmoid(&output);
        (hidden1_out, hidden2_out, hidden3_out, hidden4_out, output_out)
    }

    fn backward(&mut self, inputs: &Array2<f32>, hidden1_out: &Array2<f32>, hidden2_out: &Array2<f32>, hidden3_out: &Array2<f32>, hidden4_out: &Array2<f32>, output_out: &Array2<f32>, target: &Array2<f32>) {
        //calculate and propagate the error from last layer to first layer
        let output_error = target - output_out;
        let output_delta = output_error * NeuralNetwork::sigmoid_derivative(output_out);
        let hidden4_error = output_delta.dot(self.weights_layer4_to_output.t());
        let hidden4_delta = hidden4_error * NeuralNetwork::sigmoid_derivative(hidden4_out);
        let hidden3_error = hidden4_delta.dot(self.weights_layer3_to_layer4.t());
        let hidden3_delta = hidden3_error * NeuralNetwork::sigmoid_derivative(hidden3_out);
        let hidden2_error = hidden3_delta.dot(self.weights_layer2_to_layer3.t());
        let hidden2_delta = hidden2_error * NeuralNetwork::sigmoid_derivative(hidden2_out);
        let hidden1_error = hidden2_delta.dot(self.weights_layer1_to_layer2.t());
        let hidden1_delta = hidden1_error * NeuralNetwork::sigmoid_derivative(hidden1_out);
        //new_weights = old_weights + learning_rate * (activations_from_previous_layer.transpose() * delta_of_current_layer)
        self.weights_input_to_layer1 += inputs.t().dot(&hidden1_delta) * self.learning_rate;
        self.weights_layer1_to_layer2 += hidden1_out.t().dot(&hidden2_delta).mapv(|x| x * self.learning_rate);
        self.weights_layer2_to_layer3 += hidden2_out.t().dot(&hidden3_delta).mapv(|x| x * self.learning_rate);
        self.weights_layer3_to_layer4 += hidden3_out.t().dot(&hidden4_delta).mapv(|x| x * self.learning_rate);
        self.weights_layer4_to_output += hidden4_out.t().dot(&output_delta).mapv(|x| x * self.learning_rate);
    }


    fn train(&mut self, input: &Array2<f32>, target: &Array2<f32>, epochs: usize) {
        let n = input.nrows();
        let mut correct = 0;
        for epoch in 0..epochs {
            let (hidden1_out, hidden2_out, hidden3_out, hidden4_out, output_out) = self.forward(input);
            for i in 0..=n {

            }
            
            
        }
    }

    fn normalize_data(&self, data: &Array2<f32>) -> Array2<f32> {
        let max_price = data.iter().map(|x| x.close).fold(0.0, f64::max);
        let max_column = data.iter().map(|x| x.volume).fold(0.0, f64::max);
        let mut x_feature = Vec::new();
        let mut y_target= Vec::new();
        


}

fn main() {
    let cleankline = processed_csv_file("klines.csv");
}
