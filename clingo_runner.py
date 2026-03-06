import clingo
from pathlib import Path

asp_program_path = Path("asp_program_ex.lp")
asp_program = ""
with open(asp_program_path, "r") as f:
    asp_program = f.read()

# Collect clingo error/warning messages (includes line:col positions)
clingo_messages = []
clingo_models = []


def clingo_logger(code, message):
    clingo_messages.append((code, message))
    print(f"[clingo {code}] {message}")


ctl = clingo.Control(logger=clingo_logger)
# Try to parse and ground the program, display errors if any
try:
    ctl.add("base", [], asp_program)
    ctl.ground([("base", [])])
except RuntimeError as e:
    print(f"\nParsing/grounding failed: {e}")
    if clingo_messages:
        print("\nAll clingo messages:")
        for code, msg in clingo_messages:
            print(f"  [{code}] {msg}")
    raise

# Solve the program
ctl.solve(on_model=lambda model: clingo_models.append(model.symbols(atoms=True)))

print("\nAll models:")
for model in clingo_models:
    print(model)
