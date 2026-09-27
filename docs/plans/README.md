# Implementation plans

Code-oriented plans: which features get built, in what order, with what interfaces and tests. Each plan is a self-contained brief that an implementer (human or agent) can execute without the design discussion in their head. The *why* lives in [docs/design](../design/README.md); plans link to it and never re-argue it.

Conventions:

- One file per plan, numbered in execution order: `NN-slug.md`.
- A plan is written and merged on its own branch (`plan/<slug>`) before implementation starts on a feature branch (`feat/<slug>`).
- A plan states its goal, deliverables, what is out of scope, the steps in test-driven order, and a definition of done.
- Plans that need a design decision stop and ask; they do not decide.
- When a plan is fully implemented and merged, its file moves to [archive/](archive/README.md).
