# Validation Report

Validation date: 2026-08-31

## Automated checks

Each PDF was checked for:

- a valid `%PDF-` header;
- successful strict reopening with `pypdf`;
- non-zero page count;
- absence of encryption;
- first-page text matching the expected paper title and Dan Alistarh authorship;
- stable byte count and SHA-256 digest;
- successful text extraction for evidence review.

Result: 8 of 8 PDFs passed.

## Visual check

The first page of every PDF was rendered with Poppler and inspected in a contact sheet. Titles, authors, venue/version markings, and page layout were visually consistent with the selected records. No HTML error page, truncated download, or identity mismatch was found.

## Link check

The master guide, inventory, and all eight per-paper notes use relative links so the complete folder can be moved without breaking its internal navigation. The final audit checked 27 local Markdown links across 12 Markdown files: 0 broken links. The folder contains 8 PDFs and 0 partial-download files.

## Version boundary

- The folder keeps one official published artifact per selected paper.
- OPTQ is the title printed in the ICLR 2023 paper; GPTQ is the name commonly used by the public implementation and later literature. The note makes this naming boundary explicit.
- No arXiv draft is silently substituted for a conference version.
