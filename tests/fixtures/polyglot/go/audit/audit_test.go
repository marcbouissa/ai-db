package audit

import (
	"testing"

	"polyglot/billing"
)

func TestReviewTotal(t *testing.T) {
	log := NewAuditLog()
	total := log.ReviewTotal(100.0)
	if total != 120.0 {
		t.Fatalf("expected 120.0, got %f", total)
	}
}

func TestRound2(t *testing.T) {
	if billing.Round2(1.005) != 1.01 {
		t.Fatal("bad rounding")
	}
}
