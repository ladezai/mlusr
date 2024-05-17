use std::collections::HashSet;
use rand::prelude::*;

pub struct UniquesInAStream {
    X      : HashSet<u64>,
    rng    : ThreadRng,
    p      : f64,
    stream_size : f64,
    eps : f64,
    delta : f64
}

impl UniquesInAStream {
    pub fn new(stream_size : usize, eps : f64, delta : f64) -> Self {
        UniquesInAStream {
            X : HashSet::new(),
            rng : rand::thread_rng(),
            stream_size : stream_size as f64,
            p : 1.0,
            eps : eps, 
            delta : delta,
        }
    }

    pub fn p(&self) -> f64 {
        self.p
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }
    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn stream_size(&self) -> f64 {
        self.stream_size
    }

    pub fn thresh(&self) -> usize {
        ((12.0 / (self.eps() * self.eps()) ) * ((8.0 * self.stream_size)/self.delta()).ln()).ceil() as usize
    }

    // Implememts Algorithm 1 of https://arxiv.org/abs/2301.10191.
    pub fn update(&mut self, new_elem : &u64) -> () {
        // Same as the described algorithm
        self.stream_size = self.stream_size() + 1.0;
        
        self.X.remove(&new_elem);
        let r : f64 = self.rng.gen();
        if r <= self.p() {
            self.X.insert(*new_elem);
        }

        let t = self.thresh();
        if self.X.len() == t {
            // rand::random() generates a bool value uniformly distributed
            // NOTE: This is quite slow
            self.X.retain(|_| rand::random());

            self.p = self.p() / 2.0;
            if self.X.len() == t {
                panic!("This should not happen!");
            }
        }
    }

    pub fn to_result(&self) -> u64 {
        (self.X.len() as f64 / self.p) as u64
    }
}


fn main() {
    let example_stream : Vec<u64> = (0.. 100).map(|v| (1+(-1 as i32).pow(v)) as u64).collect();
    let mut uniques = UniquesInAStream::new(0_usize, 0.1, 0.1);
    example_stream.iter().for_each(|v| uniques.update(v));
    println!("The sequence 0,2,0,2... has 2 distinct elements? Result: {:?}", uniques.to_result());

    let example_stream : Vec<u64> = (0.. 1000000).collect();
    let mut uniques = UniquesInAStream::new(0_usize, 0.1, 0.1);
    example_stream.iter().for_each(|v| uniques.update(v));
    println!("The sequence 0,1,... 10^6-1 has 10^6 distinct elements? Result: {:?}", uniques.to_result());
}
