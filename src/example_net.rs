#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{Result, Device, Tensor, DType, D};
use candle_nn::{Module, Optimizer, Activation, Linear, VarBuilder, VarMap, linear};

use crate::sequential_layers::{Sequential};

// Plots
use plotly::common::{Mode, color::NamedColor};
use plotly::{Plot, Scatter};


pub struct Dataset { 
    train_data : Tensor,
    train_label : Tensor,
    test_data : Tensor,
    test_label : Tensor,
}

pub struct ActivizedLayer {
    layer : Linear,
    activation : Activation
}

impl ActivizedLayer {
    fn new(inp_size : usize, 
            out_size : usize, 
            activation : Activation, 
            vs : &VarBuilder) -> Result<ActivizedLayer> { 

        let layer = linear(inp_size, out_size, 
                        vs.pp(format!("i{inp_size}o{out_size}")))?;
        Ok(ActivizedLayer {
            layer : layer ,
            activation : activation
        })
    }
}

impl Module for ActivizedLayer {
    fn forward(&self, x : &Tensor) -> Result<Tensor> {
        let res = self.layer.forward(x)?;
        self.activation.forward(&res)
    }
}

// NOTE: Loss types are just special Modules.
pub trait Loss {
    fn loss(self, input : &Tensor, target : &Tensor) -> Result<Tensor>;
}

#[derive(Clone, Debug)]
pub struct MSELoss;

impl Loss for MSELoss {
    fn loss(self, input : &Tensor, target : &Tensor) -> Result<Tensor> {
        (input - target)?.sqr()?.mean_all()
    }
}

pub fn train(m: Dataset, 
                    model: impl Module,
                    mut optimizer: impl Optimizer, 
                    loss : impl Loss + Clone,
                    epochs : usize, 
                    dev: &Device) -> Result<impl Module> {
    // Export from dataset
    let train_results = m.train_data.to_device(dev)?;
    let train_votes   = m.train_label.to_device(dev)?;
    let test_votes    = m.test_data.to_device(dev)?;
    let test_results  = m.test_label.to_device(dev)?;
    // accuracy
    let mut final_accuracy: f32 = 0.0;
    
    // Training loop
    for epoch in 1..epochs + 1 {
        // Inference
        let out = model.forward(&train_votes)?;
        let loss_train = loss.clone().loss(&out, &train_results)?;
        // Optimize 
        optimizer.backward_step(&loss_train)?;

        let test_forward = model.forward(&test_votes)?;
        let loss_test = loss.clone().loss(&test_forward, &test_results)?;
        println!("Epoch: {epoch:3} Train loss: {:8.5} Test loss: {:8.5}",
                 loss_train.to_scalar::<f32>()?,
                 loss_test.to_scalar::<f32>()?
        ); 
    }
    Ok(model)
}

pub fn SIN_DATASET_EXAMPLE() -> Result<()> {
    // Sin regression
    let dev    = Device::cuda_if_available(0)?;
    let train_data  = Tensor::randn(0_f32, 6_f32, (100, 1), &dev)?;
    let train_label = train_data.sin()?;
    let test_data   = Tensor::randn(2_f32, 7_f32, (200, 1), &dev)?;
    let test_label = test_data.sin()?;
    let dt = Dataset { train_data : train_data, 
                    train_label : train_label, 
                    test_data : test_data.clone(),
                    test_label : test_label.clone() };
    let varmap = VarMap::new();
    let vs     = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let unpk1 = ActivizedLayer::new(1, 20, Activation::Relu, &vs)?;
    let unpk2 = ActivizedLayer::new(20, 40, Activation::Relu, &vs)?;
    let unpk3 = ActivizedLayer::new(40, 1, Activation::Sigmoid, &vs)?;
    let seq_net = Sequential::new(vec!(unpk1, 
                    unpk2, 
                    unpk3))?;
    let mut optim = candle_nn::SGD::new(varmap.all_vars(), 0.05)?;
    let loss = MSELoss {};
    let trained_model = train(dt, seq_net, optim, loss, 100_usize, &dev)?;
    
    let mut sorted_vec = test_data.clone().to_vec2::<f32>()?
                        .into_iter()
                        .map(|r| r[0])
                        .collect::<Vec<f32>>();
    sorted_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sorted_test_data = Tensor::new(sorted_vec, &dev)?.reshape((200,1))?;
    let pred_outs = trained_model.forward(&sorted_test_data)?;
    let true_outs = sorted_test_data.sin()?;
    draw_plot(&sorted_test_data, &pred_outs, &true_outs);
    Ok(())
}

fn draw_plot(x : &Tensor, y : &Tensor, z : &Tensor) -> Result<()> {
    // Define some sample data
    let x_values  : Vec<f32> = x.to_vec2::<f32>()?
                                .into_iter()
                                .map(|r| r[0])
                                .collect();
    let y_values  : Vec<f32> = y.to_vec2::<f32>()?
                                .into_iter()
                                .map(|r| r[0])
                                .collect();
    let z_values  : Vec<f32> = z.to_vec2::<f32>()?
                                .into_iter()
                                .map(|r| r[0])
                                .collect();
    
    // Create a scatter plot with two traces
    let trace1 = Scatter::new(x_values.clone(), y_values.clone())
        .mode(Mode::Lines)
        .name("Estimates")
        .line(plotly::common::Line::new().color(NamedColor::Black).width(5 as f64).dash(plotly::common::DashType::Solid));
    let trace2 = Scatter::new(x_values.clone(), z_values.clone())
        .mode(Mode::Lines)
        .name("Correct results")
        .line(plotly::common::Line::new().color(NamedColor::Red).width(5 as f64).dash(plotly::common::DashType::Solid));


    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    // Show the plot
    plot.show();

    Ok(())
}
