"""Transaction costs, market impact, financing and capacity.

Milestone 3. This module decides whether the platform is useful or decorative.
Flat basis-point costs are forbidden as a default anywhere (spec section 13); the
default is the square-root impact law with separate temporary and permanent
components, plus spread, financing and borrow.
"""

from __future__ import annotations

__all__: list[str] = []
