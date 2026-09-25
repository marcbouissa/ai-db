package audit

import (
	"fmt"

	"polyglot/billing"
)

// AuditLog records money movements for later review.
type AuditLog struct {
	events []string
}

// NewAuditLog builds an empty audit log.
func NewAuditLog() *AuditLog {
	return &AuditLog{events: make([]string, 0)}
}

// Record appends a timestamped event.
func (l *AuditLog) Record(event string) {
	l.events = append(l.events, fmt.Sprintf("%s %s", "now", event))
}

// ReviewTotal recomputes a total and records it.
func (l *AuditLog) ReviewTotal(amount float64) float64 {
	calc := billing.NewTaxCalculator()
	total := calc.TotalWithTax(amount)
	l.Record(fmt.Sprintf("reviewed %f", total))
	return total
}
