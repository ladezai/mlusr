use std::collections::HashMap;
use candle_core::{Result, Tensor};
use candle_nn::{Module};

#[derive(Clone, Debug)]
pub struct Sequential<T : Module> {
    layers : Vec<T>
}

impl<T : Module> Sequential<T> {
    pub fn new(layers : Vec<T>) -> Result<Sequential<T>>{
        // filters out possible errors, if any, return error
        Ok(Self { layers })
    }

}

impl<T : Module> Module for Sequential<T> {
    fn forward(&self, x: &Tensor) -> Result<Tensor>  {
        self.layers.iter()
            .try_fold(x.clone(), |acc, l| l.forward(&acc))
    }
}


///
/// Basically adds the effects according to the skip map.
///
/// Example:
/// skips = { 0 => 1, 1 => 2 }
/// 3 linear layers, L_0, L_1, L_2
///        .--- . .----.
///        |    | |    |
///        |    v |    v
/// in -> L0 ->  L1 -> L2 -> out
///
/// > let v1 = L0(in);
/// > let v2 = L1(v1) + v1;
/// > let v3 = L2(v2) + v2;
/// > out = v3
///
#[derive(Clone, Debug)]
pub struct LinearSkipConnection<T : Module> {
    layers : Vec<T>,
    skips  : HashMap<usize, usize>, 
}

impl<T : Module> LinearSkipConnection<T> {
    pub fn new(layers : Vec<T>, skips : HashMap<usize, usize>) -> Self {
        LinearSkipConnection { layers, skips }
    }
}

impl<T : Module> Module for LinearSkipConnection<T> {
    fn forward(&self, x:&Tensor) -> Result<Tensor> {
        // the idea is that skips says that from index 
        // layers[skips[i][0]] is being added to layers[skips[i][1]]
        // and so on.
        let mut hidden_state : Tensor = x.clone();
        let mut skip_states : HashMap<usize, Tensor> = HashMap::with_capacity(self.skips.len()); 
        
        // Iterate throughout the layers, the idea is to append
        // any skip to be added to the hashmap skip_states, then
        // compute the forward step in the usual manner
        for (i, layer) in self.layers.iter().enumerate() {
            // Compute previous effects from skip connections
            hidden_state = layer.forward(&hidden_state)?;            

            // Adds the effects of the residuals to the current hidden state
            // if any residual has to be added.
            if skip_states.contains_key(&i) {
                // updates the hidden state
                let e = skip_states.get(&i).unwrap();  
                hidden_state = (e + &hidden_state)?; 
                // Remove wasteful data
                skip_states.remove(&i);
            } 

            //  Adds the skip states to be updated 
            if self.skips.contains_key(&i) {
                let _val = *self.skips.get(&i).unwrap();
                // if the key value already exists, modify the Result<Tensor> value
                // otherwise adds the value hidden_state (i.e. the current state 
                // of the inference).
                if let Some(r) = skip_states.get(&_val) {
                    let vv = (r + &hidden_state)?;
                    skip_states.entry(_val).and_modify(|t| *t = vv);
                } else {
                    skip_states.insert(_val, hidden_state.clone());
                };
            }
        }

        Ok(hidden_state)
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use candle_core::{Device, Result, Tensor, DType};
    use candle_nn::{Linear, Module, VarBuilder, VarMap, linear};

    use crate::sequential_layers::{Sequential, LinearSkipConnection};
    //use sequential_layers::{Sequential, LinearSkipConnection};


    #[test]
    fn test_sequential() -> Result<()> {
        let dev    = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vs     = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let unpk1 = linear(2, 2, vs.clone()).unwrap();
        let unpk2 = linear(2, 2, vs.clone()).unwrap(); 
        let unpk3 = linear(2, 2, vs.clone()).unwrap();        
        // DEBUG unpacked VALUES
        //let trivial_tensor = Tensor::new(&[-1.0f32, -1.0], &Device::Cpu);
        //let maybe_biases = match unpacked.bias().ok_or(&trivial_tensor)
            //{
                //Ok(x) => x,
                //_y  => return Ok(())
            //};
        //println!("{:?}", unpacked.weight().to_vec2::<f32>()?);
        //println!("{:?}", (maybe_biases).to_vec1::<f32>()?);

        let layers = vec!(unpk1.clone(), unpk2.clone(), unpk3.clone());
        let net    = Sequential::new(layers)?;
        // TODO: add more points, sample them randomly
        let input  = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
        let out    = net.forward(&input)?;
        let linear_out = unpk3.forward(&unpk2.forward(&unpk1.forward(&input)?)?)?;
        
        let maybe_eq = (linear_out - out)?.abs()?.sum(0)?.sum(0)?;
        println!("{:?}", maybe_eq); 
        assert!(maybe_eq.to_scalar::<f32>()? < 1e-3_f32);

        Ok(())
    }

    #[test]
    fn test_skip_connection() -> Result<()> {
        // ''Prelude'' for generating a non trivial neural network
        let dev    = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vs     = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let unpk1 = linear(2, 2, vs.clone()).unwrap();
        let unpk2 = linear(2, 2, vs.clone()).unwrap(); 
        let unpk3 = linear(2, 2, vs.clone()).unwrap();  

        // TEST LinearSkipConnection 
        // works as expected
        // TODO: sample random inputs 
        let input  = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
        let mut skips : HashMap<usize, usize> = HashMap::new();
        skips.insert(0_usize, 2_usize);
        skips.insert(1_usize, 2_usize);
        let layers_skip_net = vec!(unpk1.clone(), unpk2.clone(), unpk3.clone());
        let skip_net = LinearSkipConnection::<Linear>::new(layers_skip_net, skips); 
        let skipper_res = skip_net.forward(&input)?;
        println!("{:?}", skipper_res.to_vec2::<f32>()?); 
        let v1 = unpk1.forward(&input)?;
        let v2 = unpk2.forward(&v1)?; 
        let v3 = (v1 + v2.clone() + unpk3.forward(&v2)?)?;
        println!("{:?}", v3.to_vec2::<f32>()?);
        let maybe_eq1 = (v3 - skipper_res)?.abs()?.sum(0)?.sum(0)?;
        println!("{:?}", maybe_eq1); 
        assert!(maybe_eq1.to_scalar::<f32>()? < 1e-3_f32);

        // Tests another type of jumps  
        let mut skips2 : HashMap<usize, usize> = HashMap::new();
        skips2.insert(0_usize, 1_usize);
        skips2.insert(1_usize, 2_usize);
        let layers_skip_net2 = vec!(unpk1.clone(), unpk2.clone(), unpk3.clone());
        let skip_net2 = LinearSkipConnection::<Linear>::new(layers_skip_net2, skips2);
            
        let skipper_res2 = skip_net2.forward(&input)?;
        println!("{:?}", skipper_res2.to_vec2::<f32>()?);
        
        let v21 = unpk1.forward(&input)?;
        let v22 = (v21.clone() + unpk2.forward(&v21)?)?; 
        let v23 = (v22.clone() + unpk3.forward(&v22)?)?;
        println!("{:?}", v23.to_vec2::<f32>()?);
        let maybe_eq2 = (v23 - skipper_res2)?.abs()?.sum(0)?.sum(0)?;
        println!("{:?}", maybe_eq2); 
        assert!(maybe_eq2.to_scalar::<f32>()? < 1e-3_f32);

        Ok(())
    }
}
