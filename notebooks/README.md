# Notebooks

**Notebooks are for exploring. Anything that produces a number you might act on
lives in `src/` with a test.**

This is not a style preference. A notebook's execution order is invisible in its
output, its intermediate state survives edits, and nothing about it is reviewed or
regression-tested. A backtest result computed in a cell is a result nobody can
reproduce, including you, next month.

Practical rules:

- Import from `quantlab`; do not redefine logic here. The `src` tree is mounted
  read-only inside the research container precisely so a notebook cannot quietly
  edit the library it is supposed to be exercising.
- When a cell becomes something you rely on, move it into `src/quantlab/` and give
  it a test. Then import it back.
- Outputs are gitignored. Commit the code, not the pictures.
