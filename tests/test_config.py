"""What `Settings` has to be true of itself (D10, review round on PR #10).

`config.py` is the only reader of the environment, so a defect in its *shape* —
as opposed to in a value — is one nothing else can catch. This file holds the
checks that are about the class rather than about any setting in it.
"""

from __future__ import annotations

import ast
import collections
import inspect

from shopagent.config import Settings


def test_no_setting_is_declared_twice():
    """A second declaration silently wins, and nothing anywhere says so.

    `langfuse_public_key`, `langfuse_secret_key` and `langfuse_host` were each
    declared twice in the class body — once with the paragraph explaining them
    and once in a bare block left over from the day they were stubbed. Python
    keeps the later one, so the documented declaration was the dead one, and
    the two could have drifted in default or type with nothing to notice.

    Nothing behaved wrongly, which is the reason this is a test rather than a
    fixed bug: the duplicate had identical values, so the whole cost was two
    sources of truth waiting to disagree. Found by review on PR #10.

    The AST rather than `model_fields`, which is exactly what cannot see this:
    Pydantic has already collapsed the duplicate by the time it exists.
    """
    tree = ast.parse(inspect.getsource(Settings))
    (klass,) = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    declared = collections.defaultdict(list)
    for statement in klass.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            declared[statement.target.id].append(statement.lineno)

    twice = {name: lines for name, lines in declared.items() if len(lines) > 1}
    assert not twice, (
        "these settings are declared more than once in Settings, and the later "
        f"declaration is the one Python keeps: {twice}"
    )
