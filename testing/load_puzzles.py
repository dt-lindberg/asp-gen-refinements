"""Loads (training) puzzles from the dataset"""

from pathlib import Path

PUZZLE_GEN_PROMPT = """---
TASK: Generate an Answer Set Program (ASP) that models and solves the given puzzle.
* Include comments to explain your reasoning.
* Occam's razor applies; always opt for the simplest solution.
"""


def data_gen(dataset_name="train", num_data=50) -> list[str]:
    """
    Args:
        num_data (int): the maximum number of data to evaluate
        dataset_name (str): a string in {'train', 'test', 'test_HA'} denoting 3 datasets
    """
    if dataset_name == "train":
        data_path = Path(
            "LogicGridPuzzleData/Train_50/annoated/annotated_Train_[i].txt"
        )
        solution_path = Path("LogicGridPuzzleData/Train_50/Solution/sol_Train_[i].txt")
    elif dataset_name == "test":
        data_path = Path("LogicGridPuzzleData/Test_50/annotated/annotated_Test_[i].txt")
        solution_path = Path("LogicGridPuzzleData/Test_50/Solution/sol_Test_[i].txt")
    else:
        data_path = Path("LogicGridPuzzleData/Test_2_50_HA/HA_[i].txt")
        solution_path = "LogicGridPuzzleData/Test_2_50_HA/sol_HA_[i].txt"

    puzzles = []  # a list of (story, constraints, constants, solution)

    for i in range(1, num_data + 1):
        i = str(i)
        puzzle_i_path = ".." / Path(str(data_path).replace("[i]", i))
        with open(puzzle_i_path, "r") as f:
            content = f.read().split("###")[0]
            content = content + PUZZLE_GEN_PROMPT
            puzzles.append(content)

    return puzzles


def format_puzzles_vllm(puzzles: list[str]):
    """Packages the puzzles into a list of 'messages' for vLLM to process in a batch"""
    messages = []
    for puzzle in puzzles:
        messages.append(
            {
                "role": "user",
                "content": puzzle,
            }
        )
    return messages


if __name__ == "__main__":
    puzzles = data_gen(dataset_name="train", num_data=3)
    for puzzle in puzzles:
        print(puzzle)
