import os

MAX_RETRIES = 3

class Widget:
    """A widget."""

    def render(self, width):
        return " " * width

def compute_total(items):
    """Sum the items."""
    return sum(items)
