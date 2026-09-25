//! Tests for the audit trail.

use crate::audit::audit::AuditLog;
use crate::billing::tax::round2;

#[test]
fn review_total_is_recorded() {
    let mut log = AuditLog::new();
    let total = log.review_total(100.0);
    assert!((total - 120.0).abs() < 0.001, "expected 120.0, got {}", total);
}

#[test]
fn rounding_is_two_places() {
    assert!((round2(1.005) - 1.01).abs() < 0.0001);
}
