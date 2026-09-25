export interface LedgerEntry {
  id: string;
  amount: number;
  currency: string;
}

export class Ledger {
  private entries: LedgerEntry[] = [];

  add(entry: LedgerEntry): void {
    this.entries.push(entry);
  }

  entriesFor(currency: string): LedgerEntry[] {
    return this.entries.filter((e) => e.currency === currency);
  }
}
