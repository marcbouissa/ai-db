# polyglot fixture

A small synthetic project spanning **TypeScript, Go and Rust**, used by
`eval/golden/polyglot_pack.jsonl` to check that `investigate` resolves symbols,
callers and tests in languages other than Python.

Each language has the same shape so the golden set can be language-symmetric:

| Concern | TypeScript | Go | Rust |
|---|---|---|---|
| tax/invoice domain | `ts/billing/tax.ts` | `go/billing/invoice.go` | `rust/billing/tax.rs` |
| supporting type | `ts/billing/ledger.ts` | — | — |
| cross-package consumer | `ts/audit/audit.ts` | `go/audit/audit.go` | `rust/audit/audit.rs` |
| test file | `ts/audit/audit.test.ts` | `go/audit/audit_test.go` | `rust/audit/audit_test.rs` |

The cross-package call is the point: `audit.reviewTotal` / `ReviewTotal` /
`review_total` calls back into the billing package, so an `impact` query has to
follow a reference **across** files and languages. Nothing here is compiled or
executed — the files exist only to be parsed.

Run the eval with `--all-files`, because the fixture is inside the repo and
would otherwise be filtered to git-tracked files only:

```bash
AI_DB_PATH=/tmp/polyglot.db ai-db --config /tmp/lex.json eval --pack \
  --golden eval/golden/polyglot_pack.jsonl \
  --root tests/fixtures/polyglot --all-files
```
