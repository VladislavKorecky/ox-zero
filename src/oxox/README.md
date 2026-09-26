# oxox

The importable package. Subpackages follow the roadmap in the top-level README and depend on each other in one direction only:

`game` ← `engine` ← `training`, with `cli` and `gui` sitting on top of whatever they need.

`game` must never import from the others.

> **Note:** this file is a placeholder describing the intended contents of the folder. Delete it once the folder has real code.
