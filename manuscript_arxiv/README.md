# arXiv preprint source

This preprint uses the scientific content and figure assets from source commit
`8e54ff7e6bf6a893ea5ea9ecdee93e560042c6a9`, verified against the official
OpenReview submission PDF (`HByw4gYKIa`). It intentionally does not include the
later scientific revisions in the other manuscript directories.

Presentation changes restore all 12 named authors and acknowledgements, use a
generic preprint style, remove the review checklist and submission comments,
and replace the anonymous code link with the public repository. Bibliographic
venue information for cited works is retained. Unicode punctuation and accented
names use portable TeX encodings. `preprintnat.bst` displays up to 20 names per
reference followed by et al.; the bibliography database retains all names.

Build from this directory with `tectonic --keep-intermediates 00_main.tex`, or
with a standard LaTeX/BibTeX workflow. The checked-in `00_main.bbl` lets arXiv
compile the complete reference list without regenerating it. Only the 15 figure
assets used by the manuscript are included.

For upload, archive the TeX, style, bibliography, BBL, and figure files with
`00_main.tex` at the archive root. Exclude this README and local build products.
