"""Example prompt for ASP generation."""

SYSTEM = "You are an expert Answer Set Program (ASP) coder."

USER = """
## Logic Grid Puzzle

Four people — Alice, Bob, Carol, and Dan — each own a different pet (cat, dog, fish, bird) \
and prefer a different drink (coffee, tea, juice, water).

Clues:
1. Alice does not own a cat or a dog.
2. The person who owns a fish drinks juice.
3. Bob drinks coffee.
4. Carol owns a dog.
5. Dan does not drink water.

---
TASK: Generate an Answer Set Program (ASP) that models and solves the puzzle.
* Include comments to explain your reasoning.
* Occam's razor applies; always opt for the simplest solution."""
