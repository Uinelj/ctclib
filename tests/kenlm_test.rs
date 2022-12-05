use std::{fs::File, path::Path};

use ctclib::{Dict, KenLM, LM};

#[test]
fn kenlm_model_works() {
    let dict = Dict::parse(File::open("data/letter.dict").unwrap()).unwrap();
    let mut kenlm = KenLM::new(&Path::new("data/overfit.arpa"), &dict).unwrap();
    let root = kenlm.start();
    let (next_state, score) = kenlm.score(&root, dict.index("M").unwrap(), dict.len());
    assert_eq!(score, -0.045306083);
    {
        let (_, score) = kenlm.finish(&next_state);
        assert_eq!(score, -2.9529781);
    }
    {
        let (next_state, _) = kenlm.score(&next_state, dict.index("I").unwrap(), dict.len());
        let (_, score) = kenlm.finish(&next_state);
        assert_eq!(score, -2.8997345);
    }
}
