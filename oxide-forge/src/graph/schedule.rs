use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decay {
    Linear,
    Exponential,
    Quadratic,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecayStage {
    pub decay: Decay,
    pub steps: usize,
    pub end_rate: f32,
}

/// Serializable sequence of continuous learning-rate stages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearningRateScheduler {
    initial_rate: f32,
    stages: Vec<DecayStage>,
    step: usize,
}

impl LearningRateScheduler {
    pub fn new(initial_rate: f32) -> Self {
        validate_rate(initial_rate, "initial learning rate");
        Self {
            initial_rate,
            stages: Vec::new(),
            step: 0,
        }
    }

    pub fn then(mut self, decay: Decay, steps: usize, end_rate: f32) -> Self {
        assert!(
            steps > 0,
            "learning-rate stage must contain at least one step"
        );
        validate_rate(end_rate, "stage end learning rate");
        if matches!(decay, Decay::Exponential) {
            assert!(
                self.final_rate() > 0.0 && end_rate > 0.0,
                "exponential learning-rate stages require positive endpoints"
            );
        }
        self.stages.push(DecayStage {
            decay,
            steps,
            end_rate,
        });
        self
    }

    pub fn linear(self, steps: usize, end_rate: f32) -> Self {
        self.then(Decay::Linear, steps, end_rate)
    }

    pub fn exponential(self, steps: usize, end_rate: f32) -> Self {
        self.then(Decay::Exponential, steps, end_rate)
    }

    pub fn quadratic(self, steps: usize, end_rate: f32) -> Self {
        self.then(Decay::Quadratic, steps, end_rate)
    }

    pub fn current(&self) -> f32 {
        self.rate_at(self.step)
    }

    pub fn current_step(&self) -> usize {
        self.step
    }

    pub fn rate_at(&self, mut step: usize) -> f32 {
        let mut start_rate = self.initial_rate;
        for stage in &self.stages {
            if step <= stage.steps {
                let progress = step as f32 / stage.steps as f32;
                return interpolate(stage.decay, start_rate, stage.end_rate, progress);
            }
            step -= stage.steps;
            start_rate = stage.end_rate;
        }
        start_rate
    }

    /// Advances after one optimizer step and returns the next step's rate.
    pub fn advance(&mut self) -> f32 {
        self.step = self.step.saturating_add(1);
        self.current()
    }

    pub fn reset(&mut self) {
        self.step = 0;
    }

    pub fn stages(&self) -> &[DecayStage] {
        &self.stages
    }

    fn final_rate(&self) -> f32 {
        self.stages
            .last()
            .map_or(self.initial_rate, |stage| stage.end_rate)
    }
}

fn interpolate(decay: Decay, start: f32, end: f32, progress: f32) -> f32 {
    match decay {
        Decay::Linear => start + (end - start) * progress,
        Decay::Exponential => start * (end / start).powf(progress),
        Decay::Quadratic => {
            let remaining = 1.0 - progress;
            end + (start - end) * remaining * remaining
        }
    }
}

fn validate_rate(rate: f32, name: &str) {
    assert!(
        rate.is_finite() && rate >= 0.0,
        "{name} must be finite and non-negative"
    );
}

#[cfg(test)]
mod tests {
    use super::LearningRateScheduler;

    #[test]
    fn linear_decay_reaches_and_holds_its_endpoint() {
        let schedule = LearningRateScheduler::new(1.0).linear(4, 0.0);
        assert_eq!(schedule.rate_at(0), 1.0);
        assert_eq!(schedule.rate_at(2), 0.5);
        assert_eq!(schedule.rate_at(4), 0.0);
        assert_eq!(schedule.rate_at(100), 0.0);
    }

    #[test]
    fn composed_stages_are_continuous() {
        let schedule = LearningRateScheduler::new(0.0)
            .linear(2, 1.0)
            .quadratic(2, 0.25)
            .exponential(2, 0.0625);
        assert_eq!(schedule.rate_at(2), 1.0);
        assert_eq!(schedule.rate_at(4), 0.25);
        assert_eq!(schedule.rate_at(6), 0.0625);
    }

    #[test]
    fn advance_tracks_optimizer_steps() {
        let mut schedule = LearningRateScheduler::new(1.0).linear(2, 0.0);
        assert_eq!(schedule.current(), 1.0);
        assert_eq!(schedule.advance(), 0.5);
        assert_eq!(schedule.current_step(), 1);
        assert_eq!(schedule.advance(), 0.0);
    }
}
