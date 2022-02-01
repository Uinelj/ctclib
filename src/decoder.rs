use std::cmp::Ordering;

#[derive(Clone, Debug, PartialEq)]
struct DecoderState {
    score: f32,
    token: i32,
    prev_blank: bool,
    am_score: f32,
    lm_score: f32,
    parent_index: isize,
}

impl DecoderState {
    fn cmp_without_score(&self, other: &DecoderState) -> Ordering {
        if self.token != other.token {
            self.token.cmp(&other.token)
        } else if self.prev_blank != other.prev_blank {
            self.prev_blank.cmp(&other.prev_blank)
        } else {
            Ordering::Equal
        }
    }

    fn cmp_without_score_then_score(&self, other: &DecoderState) -> Ordering {
        let without_score = self.cmp_without_score(other);
        if without_score != Ordering::Equal {
            without_score
        } else {
            self.cmp_by_score(other)
        }
    }

    fn cmp_by_score(&self, other: &DecoderState) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderOutput {
    score: f32,
    am_score: f32,
    lm_score: f32,
    tokens: Vec<i32>,
}

impl DecoderOutput {
    fn reserved(len: usize) -> Self {
        Self {
            score: 0.0,
            am_score: 0.0,
            lm_score: 0.0,
            tokens: vec![0; len],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderOptions {
    pub beam_size: usize,
    pub beam_size_token: usize,
    /// the decoder will ignore paths whose score is more than this value lower than the best score.
    pub beam_threshold: f32,
}

pub struct Decoder {
    options: DecoderOptions,
    /// All the new candidates that proposed based on the previous step.
    current_candidates: Vec<DecoderState>,
    current_best_score: f32,
    current_candidate_pointers: Vec<usize>,
    /// blank_index is the index of the blank token.
    blank: i32,
    /// hypothesis for each time step.
    hypothesis: Vec<Vec<DecoderState>>,
}

impl Decoder {
    pub fn new(options: DecoderOptions, blank: i32) -> Self {
        Self {
            options,
            current_candidates: Vec::new(),
            current_best_score: f32::MIN,
            current_candidate_pointers: Vec::new(),
            blank,
            hypothesis: Vec::new(),
        }
    }

    pub fn decode(&mut self, data: &[f32], steps: usize, tokens: usize) -> Vec<DecoderOutput> {
        self.decode_begin();
        self.decode_step(data, steps, tokens);
        self.decode_end();
        self.get_all_hypothesis(steps)
    }

    fn decode_begin(&mut self) {
        self.reset_candidate();
        // TODO: Compute the LM initial score.
        self.hypothesis.clear();
        self.hypothesis.push(Vec::new());
        self.hypothesis[0].push(DecoderState {
            score: 0.0,
            token: self.blank,
            prev_blank: false,
            am_score: 0.0,
            lm_score: 0.0,
            parent_index: -1 /* ROOT */,
        });
    }

    fn decode_step(&mut self, data: &[f32], steps: usize, tokens: usize) {
        // Reserve hypothesis buffer.
        while self.hypothesis.len() < steps + 2 {
            self.hypothesis.push(Vec::new());
        }

        // Loop over time steps.
        let mut target_index = (0..tokens).collect::<Vec<_>>();
        for t in 0..steps {
            if tokens > self.options.beam_size_token {
                // Collect tokens with the high score at the top `beam_size_token`.
                pdqselect::select_by(&mut target_index, self.options.beam_size_token, |&a, &b| {
                    data[t * tokens + a]
                        .partial_cmp(&data[t * tokens + b])
                        .unwrap()
                        .reverse()
                });
            }
            self.reset_candidate();
            for (prev_hyp_idx, prev_hyp) in self.hypothesis[t].iter().enumerate() {
                let prev_token = prev_hyp.token;
                for &target in target_index.iter().take(self.options.beam_size_token) {
                    let token = target as i32;
                    let am_score = data[t * tokens + target];
                    let score = prev_hyp.score + am_score;

                    if token != self.blank && (token != prev_token || prev_hyp.prev_blank) {
                        // New token
                        // TODO: Compute LM Score.
                        add_candidate(
                            &mut self.current_candidates,
                            &mut self.current_best_score,
                            self.options.beam_threshold,
                            DecoderState {
                                score,
                                token,
                                prev_blank: false,
                                am_score: prev_hyp.am_score + am_score,
                                lm_score: 0.0,
                                parent_index: prev_hyp_idx as isize,
                            },
                        );
                    } else if token == self.blank {
                        // Blank
                        add_candidate(
                            &mut self.current_candidates,
                            &mut self.current_best_score,
                            self.options.beam_threshold,
                            DecoderState {
                                score,
                                token,
                                prev_blank: true,
                                am_score: prev_hyp.am_score + am_score,
                                lm_score: 0.0,
                                parent_index: prev_hyp_idx as isize,
                            },
                        );
                    } else {
                        // Extend
                        add_candidate(
                            &mut self.current_candidates,
                            &mut self.current_best_score,
                            self.options.beam_threshold,
                            DecoderState {
                                score,
                                token,
                                prev_blank: false,
                                am_score: prev_hyp.am_score + am_score,
                                lm_score: 0.0,
                                parent_index: prev_hyp_idx as isize,
                            },
                        );
                    }
                }
            }
            // Finalize candidates.
            self.finalize_candidate(t);
        }
    }

    fn decode_end(&mut self)  {
        // TODO: Compute LM Score.
    }

    fn reset_candidate(&mut self) {
        self.current_best_score = f32::MIN;
        self.current_candidates.clear();
        self.current_candidate_pointers.clear();
    }

    /// Finalize candidates at the current time step.
    /// This prunes the candidates and sort them by score.
    fn finalize_candidate(&mut self, t: usize) {
        // 1. Gather valid candidates.
        // ================================================================
        for (i, candidate) in self.current_candidates.iter().enumerate() {
            if candidate.score > self.current_best_score - self.options.beam_threshold {
                self.current_candidate_pointers.push(i);
            }
        }

        // 2. Merge same patterns.
        // ================================================================
        // Sort candidates so that the same patterns are consecutive.
        self.current_candidate_pointers.sort_by(|a, b| {
            self.current_candidates[*a].cmp_without_score_then_score(&self.current_candidates[*b])
        });
        let mut n_candidates_after_merged = 1;
        let mut last_ptr = self.current_candidate_pointers[0];
        for i in 1..self.current_candidate_pointers.len() {
            let ptr = self.current_candidate_pointers[i];
            if self.current_candidates[ptr].cmp_without_score(&self.current_candidates[last_ptr]) != Ordering::Equal{
                // Distinct pattern.
                self.current_candidate_pointers[n_candidates_after_merged] = ptr;
                n_candidates_after_merged += 1;
                last_ptr = ptr;
            } else {
                // Same pattern.
                let max_score = self.current_candidates[last_ptr].score.max(self.current_candidates[ptr].score);
                let min_score = self.current_candidates[last_ptr].score.min(self.current_candidates[ptr].score);
                self.current_candidates[last_ptr].score = max_score + libm::log1p(libm::exp(min_score as f64 - max_score as f64)) as f32;
            }
        }
        self.current_candidate_pointers.truncate(n_candidates_after_merged);

        // 3. Sort candidates.
        if self.current_candidate_pointers.len() > self.options.beam_size {
            pdqselect::select_by(&mut self.current_candidate_pointers, self.options.beam_size, |&a, &b| {
                self.current_candidates[a].cmp_by_score(&self.current_candidates[b]).reverse()
            });
        }

        // 4. Copy candidates to output.
        let output = &mut self.hypothesis[t + 1];
        output.clear();
        for &ptr in self.current_candidate_pointers.iter().take(self.options.beam_size) {
            output.push(self.current_candidates[ptr].clone());
        }
    }

    fn get_all_hypothesis(&self, final_step: usize) -> Vec<DecoderOutput> {
        println!("{:?}", self.hypothesis);
        self.hypothesis[final_step].iter().map(|hyp| {
            let mut output = DecoderOutput::reserved(final_step);
            output.score = hyp.score;
            output.am_score = hyp.am_score;
            output.lm_score = hyp.lm_score;
            let mut hyp_ = hyp;
            for i in (0..final_step).rev() {
                output.tokens[i] = hyp_.token;
                if hyp_.parent_index == -1 {
                    break;
                }
                hyp_ = &self.hypothesis[i][hyp_.parent_index as usize];
            }
            output
        }).collect()
    }
}

fn add_candidate(
    output: &mut Vec<DecoderState>,
    current_best_score: &mut f32,
    beam_threshold: f32,
    state: DecoderState,
) {
    if state.score > *current_best_score {
        *current_best_score = state.score;
    }
    if state.score > *current_best_score - beam_threshold {
        output.push(state);
    }
}

#[cfg(test)]
mod tests {
    use crate::{DecoderOptions, Decoder};

    #[test]
    fn it_works() {
        let options  = DecoderOptions {
            beam_size: 1,
            beam_size_token: 10,
            beam_threshold: f32::MAX,
        };
        let mut decoder = Decoder::new(options, 4);
        let steps = 3;
        let tokens = 4;
        let data = &[
            1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
        ];
        let outputs = decoder.decode(data, steps, tokens);
        assert_eq!(outputs, Vec::new());
    }
}