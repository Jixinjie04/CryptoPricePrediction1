use std::io::{Read};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, TimeZone};
use ndarray::{Array2, s};

//data_processing Module： 
//Handles cryptocurrency data preprocessing including loading, parsing, and transforming raw data to train test data to feed into neural network later.
mod data_processing {
    use chrono::{DateTime, Utc, TimeZone};
    use serde::{Deserialize, Serialize};
    use csv;
    use ndarray::Array2;
    
    #[derive(Debug, Deserialize, Serialize)]
    //Represents raw cryptocurrency candlestick data from CSV with string fields that need to be parsed into appropriate numeric types.
    pub struct DirtyKline {
        #[serde(rename = "Open Time")]
        pub open_time: u64,
        #[serde(rename = "Open")]
        pub open: String,
        #[serde(rename = "High")]
        pub high: String,
        #[serde(rename = "Low")]
        pub low: String,
        #[serde(rename = "Close")]
        pub close: String,
        #[serde(rename = "Volume")]
        pub volume: String,
        #[serde(rename = "Close Time")]
        pub close_time: u64,
        #[serde(rename = "Quote Asset Volume")]
        pub quote_asset_volume: String,
        #[serde(rename = "Number of Trades")]
        pub number_of_trades: u64,
        #[serde(rename = "Taker Buy Base Asset Volume")]
        pub taker_buy_base_asset_volume: String,
        #[serde(rename = "Taker Buy Quote Asset Volume")]
        pub taker_buy_quote_asset_volume: String,
        #[serde(rename = "Ignore")]
        pub ignore: String,
    }

    // Clean, typed representation of cryptocurrency price data
    #[derive(Debug, Default)]
    pub struct CleanKline {
        pub open_time: DateTime<Utc>,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
        pub close_time: DateTime<Utc>,
        pub quote_asset_volume: f64,
        pub number_of_trades: u64,
        pub taker_buy_base_asset_volume: f64,
        pub taker_buy_quote_asset_volume: f64,
        pub ignore: f64,
    }

    // Convert raw CSV record to properly typed data
    // Transforms string values to appropriate numeric types and datetime objects
    pub fn clean_record(r: DirtyKline) -> CleanKline {
        let mut c = CleanKline::default();
        // Convert timestamp to DateTime
        c.open_time = Utc.timestamp(r.open_time as i64, 0);
        // Parse all string values to appropriate numeric types
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

    // Processes a CSV file containing cryptocurrency data
    // Returns a vector of properly typed price records
    pub fn processed_csv_file(path: &str) -> Vec<CleanKline> {
        // Configure CSV reader with appropriate settings
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)
            .expect("Failed to create CSV reader");
        
        let mut clean_klines = Vec::new();
        
        // Process each record in the CSV file
        for result in rdr.deserialize::<DirtyKline>() {
            match result {
                Ok(dirty_record) => {
                    // Convert each record to properly typed data
                    let clean_record = clean_record(dirty_record);
                    clean_klines.push(clean_record);
                },
                Err(err) => {
                    println!("Error reading record: {}", err);
                    continue;
                }
            }
        }
        
        clean_klines
    }

    // Prepares time series data for neural network training
    // Creates sliding windows of features and splits into training/testing sets
    pub fn prepare_data(data: &Vec<CleanKline>, window_size: usize, test_ratio: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        // Find maximum values for normalization
        let max_open = data.iter().map(|x| x.open).fold(0.0, f64::max);
        let max_high = data.iter().map(|x| x.high).fold(0.0, f64::max);
        let max_low = data.iter().map(|x| x.low).fold(0.0, f64::max);
        let max_close = data.iter().map(|x| x.close).fold(0.0, f64::max);
        let max_column = data.iter().map(|x| x.volume).fold(0.0, f64::max);
        
        let mut feature = Vec::new();
        let mut targets = Vec::new();
        
        // Create sliding windows of features and corresponding targets
        for i in window_size..data.len() {
            let mut window_features = Vec::new();
            // Extract features from each time step in the window
            for j in i-window_size..i {
                // Normalize data to [0-1] range using maximum values for each feature
                window_features.push((data[j].open / max_open) as f32);
                window_features.push((data[j].high / max_high) as f32);
                window_features.push((data[j].low / max_low) as f32);
                window_features.push((data[j].close / max_close) as f32);
                window_features.push((data[j].volume / max_column) as f32);
            }
            feature.push(window_features);
            // Target is the next day's closing price (normalized)
            targets.push(vec![(data[i].close / max_close) as f32]);
        }
        
        // Split data into training and testing sets
        let total_samples = feature.len();
        let test_samples = (total_samples as f32 * test_ratio) as usize;
        let train_samples = total_samples - test_samples;
        
        let train_features: Vec<Vec<f32>> = feature[0..train_samples].to_vec();
        let train_targets: Vec<Vec<f32>> = targets[0..train_samples].to_vec();
        let test_features: Vec<Vec<f32>> = feature[train_samples..].to_vec();
        let test_targets: Vec<Vec<f32>> = targets[train_samples..].to_vec();
        
        // Convert to ndarray format for efficient computation
        let x_train = Array2::from_shape_vec(
            (train_features.len(), train_features[0].len()),
            train_features.into_iter().flatten().collect()
        ).expect("Failed to create train features array");
        
        let y_train = Array2::from_shape_vec(
            (train_targets.len(), train_targets[0].len()),
            train_targets.into_iter().flatten().collect()
        ).expect("Failed to create train targets array");
        
        let x_test = Array2::from_shape_vec(
            (test_features.len(), test_features[0].len()),
            test_features.into_iter().flatten().collect()
        ).expect("Failed to create test features array");
        
        let y_test = Array2::from_shape_vec(
            (test_targets.len(), test_targets[0].len()),
            test_targets.into_iter().flatten().collect()
        ).expect("Failed to create test targets array");
        
        (x_train, y_train, x_test, y_test)
    }
}

// Neural network module for time-series prediction
// Implements a feedforward neural network with backpropagation
mod neural_network {
    use crate::data_processing::CleanKline;
    use ndarray::{Array2, s};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use plotters::prelude::*;
    use chrono::{DateTime, Utc};
    use std::error::Error;

    // Neural network structure with 5 layers (input, 3 hidden, output)
    pub struct NeuralNetwork {
        pub inputs_size: usize,
        pub layer1_size: usize,
        pub layer2_size: usize,
        pub layer3_size: usize,
        pub layer4_size: usize,
        pub outputs_size: usize,
        pub learning_rate: f32,
        weights_input_to_layer1: Array2<f32>,
        weights_layer1_to_layer2: Array2<f32>,
        weights_layer2_to_layer3: Array2<f32>,
        weights_layer3_to_layer4: Array2<f32>,
        weights_layer4_to_output: Array2<f32>,
    }

    impl NeuralNetwork {
        // Creates a new neural network with randomized weights
        pub fn new(inputs_size: usize, layer1_size: usize, layer2_size: usize, layer3_size: usize, 
                   layer4_size: usize, outputs_layer: usize, learning_rate: f32) -> Self {
            // Initialize weights with small random values for better convergence
            let weights_input_to_layer1 = Array2::random((inputs_size, layer1_size), Uniform::new(-0.5, 0.5));
            let weights_layer1_to_layer2 = Array2::random((layer1_size, layer2_size), Uniform::new(-0.5,0.5));
            let weights_layer2_to_layer3 = Array2::random((layer2_size, layer3_size), Uniform::new(-0.5,0.5));
            let weights_layer3_to_layer4 = Array2::random((layer3_size, layer4_size), Uniform::new(-0.5, 0.5));
            let weights_layer4_to_output = Array2::random((layer4_size, outputs_layer), Uniform::new(-0.5, 0.5));
            
            NeuralNetwork {
                inputs_size,
                layer1_size,
                layer2_size,
                layer3_size,
                layer4_size,
                outputs_size: outputs_layer,
                learning_rate,
                weights_input_to_layer1,
                weights_layer1_to_layer2,
                weights_layer2_to_layer3,
                weights_layer3_to_layer4,
                weights_layer4_to_output
            }
        }

        // Sigmoid activation function
        pub fn sigmoid(input: &Array2<f32>) -> Array2<f32> {
            input.mapv(|x| 1.0/(1.0 + (-x).exp()))
        }

        // Derivative of sigmoid function for backpropagation
        pub fn sigmoid_derivative(input: &Array2<f32>) -> Array2<f32> {
            input.mapv(|x| x * (1.0-x))
        }

        // Forward propagation through the network
        pub fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
            // Layer 1: Input → Hidden1
            let hidden1 = input.dot(&self.weights_input_to_layer1);
            let hidden1_out = NeuralNetwork::sigmoid(&hidden1);
            
            // Layer 2: Hidden1 → Hidden2
            let hidden2 = hidden1_out.dot(&self.weights_layer1_to_layer2);
            let hidden2_out = NeuralNetwork::sigmoid(&hidden2);
            
            // Layer 3: Hidden2 → Hidden3
            let hidden3 = hidden2_out.dot(&self.weights_layer2_to_layer3);
            let hidden3_out = NeuralNetwork::sigmoid(&hidden3);
            
            // Layer 4: Hidden3 → Hidden4
            let hidden4 = hidden3_out.dot(&self.weights_layer3_to_layer4);
            let hidden4_out = NeuralNetwork::sigmoid(&hidden4);
            
            // Output layer: Hidden4 → Output
            let output = hidden4_out.dot(&self.weights_layer4_to_output);
            let output_out = NeuralNetwork::sigmoid(&output);
            
            // Return all layer activations for backpropagation
            (hidden1_out, hidden2_out, hidden3_out, hidden4_out, output_out)
        }

        // Backpropagation to update weights based on error
        pub fn backward(&mut self, inputs: &Array2<f32>, hidden1_out: &Array2<f32>, hidden2_out: &Array2<f32>, 
                        hidden3_out: &Array2<f32>, hidden4_out: &Array2<f32>, output_out: &Array2<f32>, target: &Array2<f32>) {
            // Calculate error signal for output layer
            let output_error = target - output_out;
            let output_delta = output_error * NeuralNetwork::sigmoid_derivative(output_out);
            
            // Backpropagate error through network layers
            let weights_4_to_output_t = self.weights_layer4_to_output.t();
            let hidden4_error = output_delta.dot(&weights_4_to_output_t);
            let hidden4_delta = hidden4_error * NeuralNetwork::sigmoid_derivative(hidden4_out);
            
            let weights_3_to_4_t = self.weights_layer3_to_layer4.t();
            let hidden3_error = hidden4_delta.dot(&weights_3_to_4_t);
            let hidden3_delta = hidden3_error * NeuralNetwork::sigmoid_derivative(hidden3_out);
            
            let weights_2_to_3_t = self.weights_layer2_to_layer3.t();
            let hidden2_error = hidden3_delta.dot(&weights_2_to_3_t);
            let hidden2_delta = hidden2_error * NeuralNetwork::sigmoid_derivative(hidden2_out);
            
            let weights_1_to_2_t = self.weights_layer1_to_layer2.t();
            let hidden1_error = hidden2_delta.dot(&weights_1_to_2_t);
            let hidden1_delta = hidden1_error * NeuralNetwork::sigmoid_derivative(hidden1_out);
            
            // Update weights using gradient descent
            let inputs_t = inputs.t();
            let delta_w_input_to_1 = inputs_t.dot(&hidden1_delta) * self.learning_rate;
            self.weights_input_to_layer1 += &delta_w_input_to_1;
            
            let hidden1_t = hidden1_out.t();
            let delta_w_1_to_2 = hidden1_t.dot(&hidden2_delta) * self.learning_rate;
            self.weights_layer1_to_layer2 += &delta_w_1_to_2;
            
            let hidden2_t = hidden2_out.t();
            let delta_w_2_to_3 = hidden2_t.dot(&hidden3_delta) * self.learning_rate;
            self.weights_layer2_to_layer3 += &delta_w_2_to_3;
            
            let hidden3_t = hidden3_out.t();
            let delta_w_3_to_4 = hidden3_t.dot(&hidden4_delta) * self.learning_rate;
            self.weights_layer3_to_layer4 += &delta_w_3_to_4;
            
            let hidden4_t = hidden4_out.t();
            let delta_w_4_to_output = hidden4_t.dot(&output_delta) * self.learning_rate;
            self.weights_layer4_to_output += &delta_w_4_to_output;
        }

        // Train the neural network for specified number of epochs
        pub fn train(&mut self, x_train: &Array2<f32>, y_train: &Array2<f32>, epochs: usize) {
            for epoch in 0..epochs {
                // Forward pass
                let (hidden1_out, hidden2_out, hidden3_out, hidden4_out, output_out) = self.forward(x_train);
                // Backward pass to update weights
                self.backward(x_train, &hidden1_out, &hidden2_out, &hidden3_out, &hidden4_out, &output_out, y_train);
                
                // Print progress every 10 epochs to avoid console spam
                if epoch % 10 == 0 || epoch == epochs - 1 {
                    println!("Epoch {}/{} completed", epoch + 1, epochs);
                }
            }
        }
        
        // Evaluate model performance on test data
        pub fn test(&self, x_test: &Array2<f32>, y_test: &Array2<f32>) -> f32 {
            let (_, _, _, _, predicted_output) = self.forward(x_test);
            let mut total_error = 0.0;
            
            // Calculate MSE across all test samples
            for i in 0..x_test.nrows() {
                let pred_i = predicted_output.slice(s![i, ..]);
                let target_i = y_test.slice(s![i, ..]);
                let diff = &pred_i - &target_i;
                // Calculate MSE for each test sample
                let squared_error = diff.mapv(|x| x.powi(2)).sum();
                total_error += squared_error;
            }
            
            // Return average error across all samples
            total_error / (x_test.nrows() as f32)
        }
        
        // Prepare data for neural network training and testing
        pub fn prepare_data(&self, data: &Vec<CleanKline>, window_size: usize, test_ratio: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
            // Call the data_processing module's function
            crate::data_processing::prepare_data(data, window_size, test_ratio)
        }
        
        // Visualize predictions versus actual values
        pub fn visualize_predictions(
            data: &Vec<CleanKline>, 
            predictions: &Vec<f32>, 
            actual: &Vec<f32>, 
            window_size: usize,
            test_ratio: f32
        ) -> Result<(), Box<dyn Error>> {
            // Calculate the starting index for test data
            let total_samples = data.len() - window_size;
            let train_samples = (total_samples as f32 * (1.0 - test_ratio)) as usize;
            let test_start_idx = window_size + train_samples;
            
            // Create dates for x-axis
            let dates: Vec<DateTime<Utc>> = data[test_start_idx..test_start_idx + predictions.len()]
                .iter()
                .map(|k| k.open_time)
                .collect();
            
            // Create the plot
            let root = BitMapBackend::new("price_prediction.png", (1200, 800)).into_drawing_area();
            root.fill(&WHITE)?;
            
            // Find min and max for scaling
            let min_price = actual.iter()
                .zip(predictions.iter())
                .map(|(a, p)| a.min(*p))
                .fold(f32::INFINITY, |min_val, x| min_val.min(x));
            
            let max_price = actual.iter()
                .zip(predictions.iter())
                .map(|(a, p)| a.max(*p))
                .fold(f32::NEG_INFINITY, |max_val, x| max_val.max(x));
            
            // Add margin to chart for better visualization
            let price_range = max_price - min_price;
            let y_min = min_price - 0.1 * price_range;
            let y_max = max_price + 0.1 * price_range;
            
            // Create the chart
            let mut chart = ChartBuilder::on(&root)
                .caption("BTC Price Prediction", ("sans-serif", 40).into_font())
                .margin(10)
                .x_label_area_size(50)
                .y_label_area_size(60)
                .build_cartesian_2d(
                    dates[0]..dates[dates.len()-1],
                    y_min..y_max
                )?;
            
            // Configure grid
            chart.configure_mesh()
                .y_desc("Price (Normalized)")
                .x_desc("Date")
                .draw()?;
            
            // Draw actual prices
            chart.draw_series(LineSeries::new(
                dates.iter().zip(actual.iter()).map(|(d, p)| (*d, *p)),
                &BLUE,
            ))?
            .label("Actual Price")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
            
            // Draw predicted prices
            chart.draw_series(LineSeries::new(
                dates.iter().zip(predictions.iter()).map(|(d, p)| (*d, *p)),
                &RED,
            ))?
            .label("Predicted Price")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
            
            // Draw legend
            chart.configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
            
            println!("Visualization saved as 'price_prediction.png'");
            Ok(())
        }
    }
}

// Main entry point for the application
fn main() {
    // Load and preprocess data
    let cleankline = data_processing::processed_csv_file("klines.csv");
    let window_size = 70;
    let test_ratio = 0.2;
    
    println!("Loaded {} data points", cleankline.len());
    
    // Create neural network with pyramid architecture
    // (larger to smaller layer sizes, ending with single output)
    let mut nn = neural_network::NeuralNetwork::new(
        350,         // Input size: 5 features × 70 days
        175,         // Hidden layer 1 size 
        80,          // Hidden layer 2 size
        50,          // Hidden layer 3 size
        40,          // Hidden layer 4 size
        1,           // Output size: price prediction
        0.01         // Learning rate
    );
    
    // Prepare training and testing data
    println!("Preparing data...");
    let (x_train, y_train, x_test, y_test) = nn.prepare_data(&cleankline, window_size, test_ratio);
    println!("Training with {} samples, testing with {} samples", x_train.nrows(), x_test.nrows());
    
    // Train the model
    println!("Training the neural network...");
    nn.train(&x_train, &y_train, 1000);
    
    // Evaluate model performance
    let mse = nn.test(&x_test, &y_test);
    println!("Test MSE: {:.6}", mse);
    
    // Get predictions for visualization
    let (_, _, _, _, predictions) = nn.forward(&x_test);
    
    // Convert to Vec<f32> for visualization
    let pred_vec: Vec<f32> = predictions.column(0).to_vec();
    let actual_vec: Vec<f32> = y_test.column(0).to_vec();
    
    // Create visualization of predictions vs actual values
    match neural_network::NeuralNetwork::visualize_predictions(&cleankline, &pred_vec, &actual_vec, window_size, test_ratio) {
        Ok(_) => println!("Successfully created visualization!"),
        Err(e) => println!("Error creating visualization: {}", e),
    }
}


