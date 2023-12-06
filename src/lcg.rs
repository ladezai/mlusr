
// Generates a sequence of pseudo-randomly generated numbers,
// using the simple technique of linear congruences. 
//
// Note: this generates a easy to detect sequence of number because 
// of the separation properties, see Marsiglia's bound.
fn lcg<const n : usize>(seed : u64, a : u64, M : u64) -> [u64; n] {
    let mut nums : [u64; n] = [seed; n];
    for i in 0..(n-1) {
        nums[i+1] = a * nums[i] % M;
    }
    nums
}

#[cfg(test)]
mod tests {
    use crate::lcg::{lcg};

    #[test]
    fn lcg_test() {
        let ex = lcg::<10>(123 as u64, 
                       3 as u64, 
                     127 as u64);
        println!("{:?}", ex);
        assert_eq!([123, 115, 91, 19, 57, 44, 5, 15, 45, 8], ex);
    }
}

