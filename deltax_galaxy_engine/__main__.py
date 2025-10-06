

"""Package entry point for `python -m deltax_galaxy_engine`.
This simply forwards to the CLI main() so you can run the engine as a module.
"""
from .cli import main as _main

if __name__ == "__main__":
    raise SystemExit(_main())