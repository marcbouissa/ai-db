import { AuditLog } from "./audit";
import { TaxCalculator } from "../billing/tax";

export function testReviewTotal(): void {
  const log = new AuditLog();
  const total = log.reviewTotal(100);
  if (total !== 120) throw new Error(`expected 120, got ${total}`);
}

export function testTaxRate(): void {
  const calc = new TaxCalculator(0.1);
  if (calc.taxFor(50) !== 5) throw new Error("bad tax");
}
