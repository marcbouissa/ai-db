import { LedgerEntry } from "./ledger";

/** Rates applied to an invoice line. */
export const STANDARD_RATE = 0.2;

export class TaxCalculator {
  constructor(private rate: number = STANDARD_RATE) {}

  /** Compute the tax owed on an amount. */
  taxFor(amount: number): number {
    return round2(amount * this.rate);
  }

  /** Total including tax. */
  totalWithTax(amount: number): number {
    return round2(amount + this.taxFor(amount));
  }
}

export function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

export function invoiceTotal(entries: LedgerEntry[]): number {
  const calc = new TaxCalculator();
  return entries.reduce((sum, e) => sum + calc.totalWithTax(e.amount), 0);
}
