

///
/// Given a long string and some fragments, returns the 
/// indexes at which the fragments can be found in the original
/// string.
///
/// long_text : a long string containing the segments
/// segments : a ref to a list of strings.
/// b : base, i.e. number of symbols used in the text. 
/// q : prime for the hash base
pub fn rabin_karp(long_text : &str, 
                  segments : &[&str], 
                  b : usize, 
                  q : usize) -> Vec<Vec<usize>> {
    // general infos
    let text_length     : usize  = long_text.len();
    let num_segments    : usize  = segments.len();
    let segment_length  : usize  = segments[0].len();

    // compute b ^ (segment_len - 1) mod q
    let b_star : usize = (1.. segment_length).fold(b, |acc, _x| acc * b % q);

    // Computes the hashes of the known segments
    let segment_hashes : Vec<usize> = segments.iter()
                                           .map(|v| rolling_hash(v, b, q))
                                           .collect();

    // mut variables of the code.
    let mut positions : Vec<Vec<usize>> = vec![Vec::new(); num_segments]; 
    let mut hash_cur_pattern : usize = rolling_hash(&long_text[..segment_length], b, q);
    let mut cur_string : String = String::from(&long_text[..segment_length]);

    for (i, char) in long_text.char_indices()
                              .skip(segment_length)
                              .take(text_length - segment_length) {

        // Search for pattern by hashing.
        for j in 0.. num_segments {
            // here short circuits prevents from wasting a double-if
            // append the position only if the hash and equality are satisfied.
           if hash_cur_pattern == segment_hashes[j] &&  
                cur_string.as_str() == segments[j] {
                positions[j].push(i-segment_length);
            }
        }

        // updates the current string to check
        let s = cur_string.remove(0);
        cur_string.push(char);

        // Updates the rolling hash
        let si  = s as usize;
        let sim = char as usize;
        hash_cur_pattern  = (b * hash_cur_pattern + sim) % q;
        hash_cur_pattern  = (hash_cur_pattern + q - (si * b_star % q)) % q;
   }

    positions
}

/// Compute hash using a weighted module sum.
/// string_to_hash : string to hash
/// b : base of the hashing
/// q : modulo of the hash (so the total size of the hashed space).
pub fn rolling_hash(string_to_hash : &str, b : usize, q : usize) -> usize
{
    string_to_hash.chars()
                  .fold(0_usize, |acc, x| (acc * b + (x as usize)) % q) 
}


#[cfg(test)]
mod tests {
    use crate::rabin_karp::{rabin_karp, rolling_hash};

    #[test]
    fn rolling_hash_test() {
        let a_val = "A".chars().next().unwrap() as usize;
        let b_val = "B".chars().next().unwrap() as usize;
        let c_val = "C".chars().next().unwrap() as usize;
        let val = rolling_hash("ABC", 3_usize, 7_usize);

        println!("A = {:?}", a_val);
        println!("B = {:?}", b_val);
        println!("C = {:?}", c_val);
        println!("HASH of ABC = {:?}", val);
        assert_eq!(val, (65 * 3 * 3 + 66 * 3 + 67 ) % 7);
    }

    #[test]
    fn rabin_karp_test() {
        let seg1 = String::from("ACG"); 
        let segments = [seg1.as_str()];
        let positions = rabin_karp("ACACACGACGATG", &segments, 4_usize, 127_usize);
        println!("{:?}", positions);

        let mut v = vec![Vec::new(); 1];
        v[0].push(4_usize);
        v[0].push(7_usize);
        assert_eq!(positions, v);
    }
} 
