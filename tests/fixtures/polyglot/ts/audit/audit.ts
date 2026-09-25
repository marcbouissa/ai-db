import { TaxCalculator } from "../billing/tax";

/** Record an audit event for a money movement. */
export class AuditLog {
  private events: string[] = [];

  record(event: string): void {
    this.events.push(`${new Date().toISOString()} ${event}`);
  }

  /** Recompute a total and record it, used by the finance review. */
  reviewTotal(amount: number): number {
    const calc = new TaxCalculator();
    const total = calc.totalWithTax(amount);
    this.record(`reviewed ${total}`);
    return total;
  }
}
