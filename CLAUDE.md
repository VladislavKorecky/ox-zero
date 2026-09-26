# oxox-zero

AlphaZero engine and analysis tooling for the OXOX board game. Project description, rules, goals, and roadmap:

@README.md

## Code Style

- Comment thoroughly. The author is learning the ML theory alongside the code, so connect implementation to theory: name the concept (e.g. PUCT, policy target, value head, Dirichlet noise), state the formula or paper reference, and explain *why* the operation exists, not just what it does.
- Don't shy away from explaining non-obvious operations (tensor reshapes, masking, normalisation, backprop details). A short "how this works" comment above a tricky block is welcome, not noise.

## Workflow

- Test-driven development. For every feature: write the failing tests first, run them to confirm they fail, then implement until they pass.
- Never write implementation code before its tests exist.

## Commands

Not yet defined. Add install, test, and run commands here once the tooling is chosen.
