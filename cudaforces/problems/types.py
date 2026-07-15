"""Problem definition types shared by the registry, seed, and judge."""

from dataclasses import dataclass, field

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-5
DEFAULT_TIME_LIMIT_MS = 5000


@dataclass(frozen=True)
class Chapter:
    id: str
    num: int
    title: str
    desc: str


@dataclass(frozen=True)
class ProblemDef:
    slug: str
    position: int
    title: str
    difficulty: str
    rating: int
    chapter: Chapter
    tags: list[str]
    summary: str
    statement: list[str]
    requirements: list[str]
    examples: list[dict[str, str]]
    constraints: list[str]
    note: str
    starter_code: str
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    time_limit_ms: int = DEFAULT_TIME_LIMIT_MS

    @property
    def module_name(self) -> str:
        """Python package directory for this problem's ref/harness assets."""
        return self.slug.replace("-", "_")


@dataclass(frozen=True)
class RefCase:
    """One judge test: named inputs in harness read order.

    Scalars (int/float) are written space-separated on line 1 of the .in file;
    array values follow, whitespace-separated, in dict order. The harness for
    the problem reads them back in exactly that order.
    """

    inputs: dict[str, object] = field(default_factory=dict)
