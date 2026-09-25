//! Tax computation for invoices.

/// Rate applied to a standard-rated invoice line.
pub const STANDARD_RATE: f64 = 0.2;

/// A single billable invoice line.
pub struct Invoice {
    pub id: String,
    pub amount: f64,
}

/// Computes tax owed on an amount.
pub struct TaxCalculator {
    rate: f64,
}

impl TaxCalculator {
    /// Build a calculator using the standard rate.
    pub fn new() -> Self {
        TaxCalculator { rate: STANDARD_RATE }
    }

    /// Tax owed on `amount`.
    pub fn tax_for(&self, amount: f64) -> f64 {
        round2(amount * self.rate)
    }

    /// Amount including tax.
    pub fn total_with_tax(&self, amount: f64) -> f64 {
        round2(amount + self.tax_for(amount))
    }
}

/// Round to two decimal places.
pub fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Sum the taxed totals of every invoice.
pub fn invoice_total(invoices: &[Invoice]) -> f64 {
    let calc = TaxCalculator::new();
    let mut total = 0.0;
    for inv in invoices {
        total += calc.total_with_tax(inv.amount);
    }
    total
}
