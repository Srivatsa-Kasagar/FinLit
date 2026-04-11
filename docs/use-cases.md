# FinLit — Business Context

A plain-language overview of what FinLit does, who uses it, and why it exists. For the developer documentation, see the [README](../README.md).

---

## The problem in one paragraph

Canadian financial institutions, fintechs, and accounting firms process large volumes of standardized documents: T4s, T5s, T4As, NR4s, SEDAR filings, bank statements. The information on these documents has to get into databases, tax forms, and compliance reports. Today, most teams either (a) pay people to type the numbers in, (b) buy a US-hosted SaaS extraction tool and accept that client SINs and CRA business numbers leave Canadian infrastructure, or (c) build their own extractor from scratch and maintain it forever. None of these are good options for regulated industries.

FinLit is option (d): an open-source extraction library, pre-built for Canadian documents, that runs entirely inside your own infrastructure — including the AI model, if you want.

---

## Who uses it

### Canadian fintechs
Lending, wealth, and budgeting apps where users upload T4s and bank statements to prove income. FinLit turns those uploads into validated, typed fields with confidence scores, so the golden path is automated and only low-confidence fields go to a human reviewer.

**Example:** A mortgage pre-qualification app ingests a user's T4 and wants `box_14_employment_income` as a float, with a trace back to the page and box it came from, in under 10 seconds.

### Banks and credit unions
Back-office pipelines that process SEDAR filings, corporate account statements, or large batches of tax slips. FinLit's `BatchPipeline` handles the parallelism, and every run produces an immutable audit log suitable for internal compliance review.

**Example:** A credit union batch-processes 40,000 T5s at year-end, exports a CSV, and keeps the audit log for seven years to satisfy internal audit.

### Accounting and tax software
Client portals where accountants or clients upload source documents and the software pre-fills CRA forms. FinLit produces the typed fields; the vendor's product logic does the mapping to form lines.

**Example:** A cloud accounting product lets a small-business owner drag a stack of T4A slips into a web upload; the extracted amounts appear in the right boxes of the T1 draft, ready for review.

### On-premises and air-gapped deployments
Federal agencies, hospitals, and OSFI-regulated institutions where no document or AI call can leave the network. FinLit's Ollama backend keeps the entire pipeline — parsing, extraction, validation, audit — inside the firewall.

**Example:** A provincial healthcare organization processes T4As for staff benefits adjudication. Nothing goes to a cloud API. The audit log is reviewed monthly.

---

## Why on-premises matters for Canadian regulated industries

- **PIPEDA** (federal privacy law) restricts cross-border transfer of personal information without adequate safeguards. Cloud extraction APIs hosted in the US create compliance friction.
- **OSFI B-10** (Third-Party Risk Management) requires financial institutions to assess and control the risk of outsourced data processing.
- **Provincial privacy laws** (Quebec's Law 25, BC's PIPA, Alberta's PIPA) add their own cross-border rules.
- **Client trust** — "your SIN never leaves our infrastructure" is a feature customers understand.

FinLit doesn't make compliance automatic; you still have to do the work. But it eliminates the data-residency question entirely: the library runs where your Python runs, and with Ollama even the AI model does too.

---

## Build vs. buy vs. FinLit

Teams that need Canadian document extraction typically face three choices. Rough comparison:

| | Build in-house | US SaaS extractor | FinLit |
|---|---|---|---|
| Up-front engineering | Months | Days (integration) | Hours |
| Ongoing maintenance | You own it | Vendor-managed | Community + your fork |
| Data residency | ✅ | ❌ Usually US | ✅ Your infrastructure |
| Canadian document schemas | You write them | Generic fields only | Pre-built (T4, T5, T4A, NR4, …) |
| Per-field confidence | You build it | Sometimes | ✅ Built in |
| Audit log | You build it | Sometimes | ✅ Built in, immutable |
| PIPEDA PII detection | You build it | Sometimes | ✅ Presidio + Canadian recognizers |
| Annual CRA spec updates | Your team tracks them | Vendor tracks them | Contributed as YAML PRs |
| Cost model | Salary + opportunity cost | Per-document fees | Free (Apache 2.0) |

The honest pitch: **FinLit is the layer every Canadian team building on these documents ends up writing anyway.** Rather than each team writing it slightly differently with no audit trail, the work is shared, open-source, and maintained by practitioners.

---

## What FinLit is not

- **Not a hosted service.** There is no `finlit.com` API, no dashboard, no SaaS tier. It's a Python library you `pip install` into your own application. If you want a managed product built on top, see [LocalMind Sovereign](https://caseonix.com/localmind).
- **Not a general document AI platform.** It's specifically tuned for Canadian financial and tax documents. For invoices, contracts, or arbitrary PDFs, tools like Docling alone or LlamaParse are better fits.
- **Not a replacement for human review.** Low-confidence fields go into a review queue by design. FinLit is built around the assumption that a human validates edge cases before the data hits a tax form or a ledger.
- **Not a compliance certification.** It gives you the audit trail and PII detection you need to demonstrate compliance; the actual compliance program is still your responsibility.

---

## How it fits into a product

FinLit is a component, not a product. A typical architecture:

```
┌──────────────┐   ┌────────────────┐   ┌─────────┐   ┌──────────────┐
│   User       │ → │   Your app's   │ → │ FinLit  │ → │  Your DB /   │
│   uploads    │   │    backend     │   │ library │   │  CRA form /  │
│   PDF        │   │   (Python)     │   │         │   │  review queue│
└──────────────┘   └────────────────┘   └─────────┘   └──────────────┘
                            │                 │
                            │                 ▼
                            │          ┌──────────────┐
                            │          │ Audit log →  │
                            │          │ compliance   │
                            │          │ archive      │
                            │          └──────────────┘
                            ▼
                   ┌──────────────┐
                   │  LLM backend │
                   │  (Claude /   │
                   │   OpenAI /   │
                   │   Ollama)    │
                   └──────────────┘
```

Your product owns the UX, the workflow, the review queue UI, and the storage. FinLit owns "bytes in, validated typed fields + confidence + audit log out."

---

## Licensing and sustainability

FinLit is licensed under Apache 2.0. You can use it in commercial products, fork it, embed it, and ship it inside closed-source applications without restriction.

It's maintained by [Caseonix](https://caseonix.com), a Waterloo-based company building document intelligence products for Canadian regulated industries. FinLit is the open-source extraction engine that sits inside Caseonix's commercial product, [LocalMind Sovereign](https://caseonix.com/localmind). Open-sourcing the engine is deliberate: the Canadian document layer is infrastructure that should be shared, and a well-maintained OSS library is more useful than a proprietary one that only one company's customers can benefit from.

If you use FinLit in production, schema contributions and bug reports are the most useful ways to give back. If you need a commercial relationship — support, integration help, or features built on your timeline — reach out via caseonix.com.

---

## Further reading

- [README](../README.md) — developer documentation, installation, API reference
- [Docling](https://github.com/docling-project/docling) — the IBM-maintained parser FinLit wraps
- [pydantic-ai](https://github.com/pydantic/pydantic-ai) — the model-agnostic LLM orchestration layer
- [Microsoft Presidio](https://github.com/microsoft/presidio) — the PII detection engine
- [LocalMind Sovereign](https://caseonix.com/localmind) — commercial product built on FinLit
