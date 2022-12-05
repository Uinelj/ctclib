/*! KenLM model bindings
*
*    This is really early work and it might crash at any error.
*    1. Instantiate a new model
*    2. Begin a context
*    3. Feed words, carrying state between calls
*    4. Send end of sentence token
*
!*/
use std::{ffi::CString, fmt::Debug, path::Path};

use crate::{Dict, LMStateRef};

use super::LM;

pub type KenLMWordIndex = ctclib_kenlm_sys::lm_WordIndex;

#[derive(Debug, Clone)]
pub struct KenLMState(ctclib_kenlm_sys::lm_ngram_State);

impl KenLMState {
    fn new() -> Self {
        Self(unsafe { std::mem::zeroed() })
    }

    fn with_ptr<T: 'static>(
        &self,
        f: impl FnOnce(*const ctclib_kenlm_sys::lm_ngram_State) -> T,
    ) -> T {
        f(&self.0 as *const _)
    }

    fn with_mut_ptr<T: 'static>(
        &mut self,
        f: impl FnOnce(*mut ctclib_kenlm_sys::lm_ngram_State) -> T,
    ) -> T {
        let ptr = &mut self.0 as *mut ctclib_kenlm_sys::lm_ngram_State;
        f(ptr)
    }
}

/// A wrapper of a KenLM Model.
pub struct Model(*mut ctclib_kenlm_sys::lm_base_Model);

unsafe impl Sync for Model {}
unsafe impl Send for Model {}

impl Model {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        if path.try_exists()? {
            let x = CString::new(path.to_str().unwrap()).unwrap();
            let model =
                unsafe { ctclib_kenlm_sys::lm_ngram_LoadVirtualWithDefaultConfig(x.as_ptr()) };
            Ok(Self(model))
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{path:?}"),
            ))
        }
    }

    /// Get the base vocabulary of the model
    pub fn vocab(&self) -> Vocabulary {
        Vocabulary(
            unsafe { ctclib_kenlm_sys::lm_base_Model_BaseVocabulary(self.0) },
            self,
        )
    }

    pub fn begin_context(&self) -> KenLMState {
        let mut state = KenLMState::new();
        state.with_mut_ptr(|ptr| unsafe {
            ctclib_kenlm_sys::lm_base_Model_BeginSentenceWrite(self.0, ptr as *mut _)
        });
        state
    }

    pub fn base_score(&self, state: &KenLMState, token: KenLMWordIndex) -> (KenLMState, f32) {
        state.with_ptr(|state_ptr| {
            let mut outstate = KenLMState::new();
            let score = outstate.with_mut_ptr(|out| unsafe {
                ctclib_kenlm_sys::lm_base_Model_BaseScore(
                    self.0,
                    state_ptr as *const _,
                    token as u32,
                    out as *mut _,
                )
            });
            (outstate, score)
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            ctclib_kenlm_sys::lm_base_Model_delete(self.0);
        }
    }
}

/// A wrapper of a reference to KenLM Vocabulary
pub struct Vocabulary<'a>(*const ctclib_kenlm_sys::lm_base_Vocabulary, &'a Model);

impl<'a> Vocabulary<'a> {
    pub fn end_sentence(&self) -> KenLMWordIndex {
        unsafe { ctclib_kenlm_sys::lm_base_Vocabulary_EndSentence(self.0) }
    }

    pub fn index(&self, x: &str) -> KenLMWordIndex {
        unsafe {
            ctclib_kenlm_sys::lm_base_Vocabulary_Index(
                self.0,
                x.as_ptr() as *const _,
                x.as_bytes().len() as u64,
            )
        }
    }
}

#[test]
fn load_model_and_get_vocab() {
    let model = Model::new(Path::new("data/overfit.arpa")).unwrap();
    let vocab = model.vocab();
    assert_eq!(vocab.end_sentence(), 2);
    assert_eq!(vocab.index("M"), 3);
    assert_eq!(vocab.index("I"), 4);

    let begin_context = model.begin_context();
    let (next_context, score) = model.base_score(&begin_context, vocab.index("M"));
    assert_eq!(score, -0.045306083);
    let (_, score) = model.base_score(&next_context, model.vocab().index("I"));
    assert_eq!(score, -0.019120596);

    // Drop explictly.
    std::mem::drop(model);
}

/// KenLM integration for ctc decoding.
/// KenLM is a n-gram language model library written in C++.
/// See https://github.com/kpu/kenlm for more details about KenLM itself.
pub struct KenLM {
    model: Model,
    idx_to_kenlm_idx: Vec<KenLMWordIndex>,
    n_vocab: usize,
}

impl KenLM {
    pub fn new(path: &Path, dict: &Dict) -> std::io::Result<Self> {
        // TODO: convert user vocabulary to KenLM's vocabulary
        let model = Model::new(path)?;
        let vocab = model.vocab();

        let mut idx_to_kenlm_idx = vec![0; dict.len()];

        for (word, &idx) in dict.iter() {
            let kenlm_idx = vocab.index(word);
            idx_to_kenlm_idx[idx as usize] = kenlm_idx;
        }

        Ok(Self {
            model,
            idx_to_kenlm_idx,
            n_vocab: dict.len(),
        })
    }

    pub fn perplexity(&self, sentence: &str) -> f32 {
        let nb_words = sentence.split_whitespace().count() as f32 + 1f32; // account for </s>

        10f32.powf(-self.sentence_score(sentence) / nb_words)
    }

    pub fn sentence_score(&self, sentence: &str) -> f32 {
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        let token_ids: Vec<_> = tokens
            .iter()
            .map(|tok| self.model.vocab().index(tok))
            .collect();
        let mut total = 0f32;

        let mut state = self.model.begin_context();
        for token_id in token_ids {
            let (new_state, score) = self.model.base_score(&state, token_id);
            total += score;
            state = new_state;
        }
        let (_, score) = self
            .model
            .base_score(&state, self.model.vocab().end_sentence());
        total + score
    }
}

impl LM for KenLM {
    type State = KenLMState;

    fn start(&mut self) -> LMStateRef<Self::State> {
        let initial_state = self.model.begin_context();
        LMStateRef::new(initial_state)
    }

    fn score(
        &mut self,
        state: &LMStateRef<Self::State>,
        token: i32,
        n_vocab: usize,
    ) -> (LMStateRef<Self::State>, f32) {
        let kenlm_idx = self.idx_to_kenlm_idx[token as usize];
        let (next_kenlm_state, score) = {
            self.model
                .base_score(&state.borrow_internal_state(), kenlm_idx)
        };
        let outstate = state.child(token, n_vocab, next_kenlm_state);
        (outstate, score)
    }

    fn finish(&mut self, state: &LMStateRef<Self::State>) -> (LMStateRef<Self::State>, f32) {
        let eos = self.model.vocab().end_sentence();
        let (next_kenlm_state, score) =
            { self.model.base_score(&state.borrow_internal_state(), eos) };
        let outstate = state.child(self.n_vocab as i32, self.n_vocab, next_kenlm_state);
        (outstate, score)
    }
}
