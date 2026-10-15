"""CLI parameter types shared by every ``peters-dome stage`` command (gaia-cli style)."""

from typing import Annotated

from cyclopts import Parameter

from .common import STORE

Input = Annotated[str, Parameter(name=["--input", "-i"])]
Start = Annotated[str, Parameter(name=["--start", "-s"])]
End = Annotated[str, Parameter(name=["--end", "-e"])]
# -o is the single container store; each dataset lands in its own --group
Output = Annotated[str, Parameter(name=["--output", "-o"])]
Group = Annotated[str, Parameter(name=["--group", "-g"])]
Overwrite = Annotated[bool, Parameter(name=["--overwrite"], negative="")]
SkipExisting = Annotated[bool, Parameter(name=["--skip-existing"], negative="")]

__all__ = ["STORE", "End", "Group", "Input", "Output", "Overwrite", "SkipExisting", "Start"]
