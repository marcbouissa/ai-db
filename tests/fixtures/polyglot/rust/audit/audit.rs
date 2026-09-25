//! Audit trail for money movements.

use crate::billing::tax::{round2, TaxCalculator};

/// Records money movements for later review.
pub struct AuditLog {
    events: Vec<String>,
}

impl AuditLog {
    /// Build an empty audit log.
    pub fn new() -> Self {
        AuditLog { events: Vec::new() }
    }

    /// Append a timestamped event.
    pub fn record(&mut self, event: &str) {
        self.events.push(format!("now {}", event));
    }

    /// Recompute a total and record it.
    pub fn review_total(&mut self, amount: f64) -> f64 {
        let calc = TaxCalculator::new();
        let total = calc.total_with_tax(amount);
        self.record(&format!("reviewed {}", round2(total)));
        total
    }
}
