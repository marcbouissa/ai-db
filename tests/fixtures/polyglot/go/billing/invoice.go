package billing

// Rates applied to an invoice line.
const StandardRate = 0.2

// Invoice is a single billable line.
type Invoice struct {
	ID     string
	Amount float64
}

// TaxCalculator computes tax owed on an amount.
type TaxCalculator struct {
	rate float64
}

// NewTaxCalculator builds a calculator with the standard rate.
func NewTaxCalculator() *TaxCalculator {
	return &TaxCalculator{rate: StandardRate}
}

// TaxFor returns the tax owed on an amount.
func (c *TaxCalculator) TaxFor(amount float64) float64 {
	return Round2(amount * c.rate)
}

// TotalWithTax returns the amount including tax.
func (c *TaxCalculator) TotalWithTax(amount float64) float64 {
	return Round2(amount + c.TaxFor(amount))
}

// Round2 rounds a value to two decimal places.
func Round2(value float64) float64 {
	return float64(int64(value*100+0.5)) / 100
}

// InvoiceTotal sums the taxed totals of every invoice.
func InvoiceTotal(invoices []Invoice) float64 {
	calc := NewTaxCalculator()
	total := 0.0
	for _, inv := range invoices {
		total += calc.TotalWithTax(inv.Amount)
	}
	return total
}
