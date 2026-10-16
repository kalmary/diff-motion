use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use crate::{config::Mode, types::SemanticMask};

#[derive(Clone, Copy)]
pub struct Logger {
    mode: Mode,
}

impl Logger {
    pub fn new(mode: Mode) -> Self {
        Self { mode }
    }

    pub fn enabled(&self) -> bool {
        self.mode != Mode::None
    }

    fn write(&self, level: &str, message: impl Display) {
        if self.enabled() {
            eprintln!("[{level}] {message}");
        }
    }

    pub fn info(&self, message: impl Display) {
        self.write("INFO", message);
    }
    pub fn warn(&self, message: impl Display) {
        if self.mode == Mode::Debug {
            self.write("WARN", message);
        }
    }
    pub fn error(&self, message: impl Display) {
        self.write("ERROR", message);
    }
    pub fn debug(&self, message: impl Display) {
        if self.mode == Mode::Debug {
            self.write("DEBUG", message);
        }
    }
}

pub fn info(message: impl Display) {
    Logger::new(Mode::default()).info(message);
}

pub fn warn(message: impl Display) {
    Logger::new(Mode::default()).warn(message);
}

pub fn error(message: impl Display) {
    Logger::new(Mode::default()).error(message);
}

#[derive(Default)]
struct Counts {
    frames: u64,
    pixels: u64,
    foreground: u64,
    ego_frames: u64,
    unreliable_ego: u64,
}

impl Counts {
    fn record(&mut self, pixels: u64, foreground: u64, ego_reliable: Option<bool>) {
        self.frames += 1;
        self.pixels += pixels;
        self.foreground += foreground;
        if let Some(reliable) = ego_reliable {
            self.ego_frames += 1;
            self.unreliable_ego += u64::from(!reliable);
        }
    }

    fn report(&self, elapsed: Duration, prefix: &str) -> Report {
        let fps = if elapsed.is_zero() {
            "n/a".to_string()
        } else {
            format!("{:.2}", self.frames as f64 / elapsed.as_secs_f64())
        };
        let (background, foreground) = if self.pixels == 0 {
            ("n/a".to_string(), "n/a".to_string())
        } else {
            (
                format!(
                    "{:.2}%",
                    (self.pixels - self.foreground) as f64 * 100.0 / self.pixels as f64
                ),
                format!(
                    "{:.2}%",
                    self.foreground as f64 * 100.0 / self.pixels as f64
                ),
            )
        };
        let mut message = format!("{prefix}fps={fps} mask_0={background} mask_1={foreground}");
        if self.unreliable_ego != 0 {
            message.push_str(&format!(
                " ego_motion_unreliable={:.2}% (new detections paused only during unreliable estimates)",
                self.unreliable_ego as f64 * 100.0 / self.ego_frames as f64
            ));
        }
        Report {
            warning: false,
            message,
        }
    }
}

pub(crate) struct Report {
    pub(crate) warning: bool,
    pub(crate) message: String,
}

impl Report {
    pub(crate) fn write(&self, logger: &Logger) {
        if self.warning {
            logger.warn(&self.message);
        } else {
            logger.info(&self.message);
        }
    }
}

pub(crate) struct StreamStats {
    started: Instant,
    interval_started: Instant,
    interval: Counts,
    total: Counts,
    last_fps: Option<f64>,
    collect_masks: bool,
}

impl StreamStats {
    pub(crate) fn new(now: Instant, collect_masks: bool) -> Self {
        Self {
            started: now,
            interval_started: now,
            interval: Counts::default(),
            total: Counts::default(),
            last_fps: None,
            collect_masks,
        }
    }

    pub(crate) fn record(&mut self, mask: &SemanticMask, ego_reliable: Option<bool>) -> u64 {
        let (pixels, foreground) = if self.collect_masks {
            (
                mask.data.len() as u64,
                mask.data.iter().filter(|&&pixel| pixel == 1).count() as u64,
            )
        } else {
            (0, 0)
        };
        self.interval.record(pixels, foreground, ego_reliable);
        self.total.record(pixels, foreground, ego_reliable);
        foreground
    }

    pub(crate) fn report(&mut self, now: Instant) -> Option<Report> {
        let elapsed = now.duration_since(self.interval_started);
        if elapsed < Duration::from_secs(1) || self.interval.frames == 0 {
            return None;
        }
        self.last_fps = Some(self.interval.frames as f64 / elapsed.as_secs_f64());
        let report = self.interval.report(elapsed, "");
        self.interval = Counts::default();
        self.interval_started = now;
        Some(report)
    }

    pub(crate) fn finish(&self, now: Instant) -> Report {
        self.total
            .report(now.duration_since(self.started), "Stream stopped: ")
    }

    #[cfg(any(feature = "display", test))]
    pub(crate) fn fps(&self) -> Option<f64> {
        self.last_fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SemanticMask;
    use std::time::{Duration, Instant};

    fn mask(data: Vec<u8>) -> SemanticMask {
        SemanticMask {
            width: data.len() as u32,
            height: 1,
            data,
        }
    }

    #[test]
    fn loggers_filter_messages_without_shared_state_or_hidden_formatting() {
        use std::cell::Cell;
        struct Message<'a>(&'a Cell<u32>);
        impl Display for Message<'_> {
            fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.set(self.0.get() + 1);
                output.write_str("message")
            }
        }
        let formatted = Cell::new(0);
        let message = Message(&formatted);
        let quiet = Logger::new(Mode::None);
        quiet.info(&message);
        quiet.warn(&message);
        quiet.error(&message);
        quiet.debug(&message);
        let info = Logger::new(Mode::Info);
        info.debug(&message);
        info.warn(&message);
        assert_eq!(formatted.get(), 0);
        let debug = Logger::new(Mode::Debug);
        debug.debug(&message);
        debug.warn(&message);
        assert_eq!(formatted.get(), 2);
        info.info(&message);
        assert_eq!(formatted.get(), 3);
        quiet.error(&message);
        assert_eq!(formatted.get(), 3);
    }

    #[test]
    fn quiet_mode_keeps_preview_fps_without_mask_statistics() {
        let start = Instant::now();
        let mut stats = StreamStats::new(start, false);
        stats.record(&mask(vec![1, 1]), None);
        let report = stats.report(start + Duration::from_secs(1)).unwrap();
        assert_eq!(stats.fps(), Some(1.0));
        assert!(report.message.contains("mask_0=n/a mask_1=n/a"));
    }

    #[test]
    fn reports_window_fps_and_weighted_mask_percentages() {
        let start = Instant::now();
        let mut stats = StreamStats::new(start, true);
        assert_eq!(stats.fps(), None);
        stats.record(&mask(vec![0, 0]), None);
        stats.record(&mask(vec![1; 6]), Some(true));
        assert!(stats.report(start + Duration::from_millis(999)).is_none());
        let report = stats.report(start + Duration::from_secs(1)).unwrap();
        assert_eq!(stats.fps(), Some(2.0));
        assert!(!report.warning);
        assert_eq!(report.message, "fps=2.00 mask_0=25.00% mask_1=75.00%");
        assert!(stats.report(start + Duration::from_secs(2)).is_none());
    }

    #[test]
    fn final_summary_includes_prior_windows_and_excludes_initial_ego_estimate() {
        let start = Instant::now();
        let mut stats = StreamStats::new(start, true);
        stats.record(&mask(vec![0, 0]), None);
        stats.record(&mask(vec![0, 1]), Some(false));
        let window = stats.report(start + Duration::from_secs(1)).unwrap();
        assert!(!window.warning);
        assert!(window.message.contains("ego_motion_unreliable=100.00%"));
        stats.record(&mask(vec![1, 1]), Some(true));
        let summary = stats.finish(start + Duration::from_millis(1500));
        assert!(!summary.warning);
        assert!(summary
            .message
            .starts_with("Stream stopped: fps=2.00 mask_0=50.00% mask_1=50.00%"));
        assert!(summary.message.contains("ego_motion_unreliable=50.00%"));
    }

    #[test]
    fn empty_and_zero_duration_summaries_do_not_divide_by_zero() {
        let start = Instant::now();
        let stats = StreamStats::new(start, true);
        let summary = stats.finish(start);
        assert!(!summary.warning);
        assert_eq!(
            summary.message,
            "Stream stopped: fps=n/a mask_0=n/a mask_1=n/a"
        );
        assert_eq!(
            stats.finish(start + Duration::from_secs(1)).message,
            "Stream stopped: fps=0.00 mask_0=n/a mask_1=n/a"
        );
    }

    #[test]
    fn reliability_summary_is_info_without_frame_counts() {
        let start = Instant::now();
        let mut stats = StreamStats::new(start, true);
        for index in 0..30 {
            stats.record(&mask(vec![0]), Some(index != 0));
        }
        let report = stats.report(start + Duration::from_secs(1)).unwrap();
        assert!(!report.warning);
        assert!(report.message.starts_with("fps=30.00 "));
        assert!(!report.message.contains("frames="));
        assert!(report.message.contains("ego_motion_unreliable=3.33%"));
        assert!(report
            .message
            .contains("paused only during unreliable estimates"));
    }
}
