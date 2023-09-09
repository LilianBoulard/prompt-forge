from __future__ import annotations

import random

from typing import Sequence
from argparse import ArgumentParser
from pathlib import Path


class Block:

    def __init__(self, name: str, parameters: dict[str, str | bool], keywords: list[str]) -> Block:
        self.name = name
        self.parameters = parameters
        self.keywords = keywords

    @classmethod
    def from_lines(cls, lines: list[str]) -> Block:
        """
        Main constructor.
        """
        name_line = lines[0]
        name, *params = name_line[1:-1].split("]")  # Trick for splitting in case the params are not specified
        if params:
            params = params[0][1:]  # Get the value and strip the opening parenthesis
            parameters = cls.parse_parameters(params)
        else:
            parameters = {}
        keywords = [line.strip() for line in lines[1:] if line.strip()]
        return cls(name.strip(), parameters, keywords)

    @staticmethod
    def parse_parameters(param_str: str) -> dict[str, str | bool]:
        parameters = {}
        for param in param_str.split(";"):
            if "=" in param:
                key, value = param.split("=")
                parameters[key.strip()] = value.strip()
            else:
                parameters[param.strip()] = True
        return parameters

    def generate_keywords(self, activated_blocks: list[str], exclusive_blocks: list[str]) -> list[str]:
        if self.name in exclusive_blocks:
            return []

        if self.parameters.get("force"):
            return self.force_generate_keywords()

        num_param = self.parameters.get("num")
        if num_param:
            min_num, max_num = map(int, num_param.split("-")) if "-" in num_param else (int(num_param), int(num_param))
        else:
            min_num, max_num = (0, 1) if self.parameters.get("optional") else (1, 1)
        num_keywords = random.randint(min_num, max_num)

        return [self.generate_keyword() for _ in range(num_keywords)]

    def force_generate_keywords(self) -> list[str]:
        return [self.generate_keyword(keyword) for keyword in self.keywords]

    def generate_keyword(self, keyword: str | None = None):
        keyword = keyword or random.choice(self.keywords)
        while "[" in keyword:
            keyword = self.resolve_brackets(keyword)
        while "(" in keyword:
            keyword = self.resolve_parentheses(keyword)
        return keyword.strip()
    
    def resolve_brackets(self, keyword):
        while "[" in keyword:
            start = keyword.rfind("[")
            end = keyword.find("]", start) + 1
            choices = keyword[start + 1:end - 1].split("|")
            choice = random.choice(choices).strip()
            keyword = keyword[:start] + choice + keyword[end:]
        return keyword

    def resolve_parentheses(self, keyword):
        while "(" in keyword:
            start = keyword.rfind("(")
            end = keyword.find(")", start) + 1
            choices = keyword[start + 1:end - 1].split("|") + [""]
            choice = random.choice(choices).strip()
            keyword = keyword[:start] + choice + keyword[end:]
        return keyword


class Generator:

    """
    A generator created from a config.
    Capable of creating new prompts on demand and with no overhead once built.
    """

    def __init__(self, blocks: Sequence[Block]) -> Generator:
        self.blocks = blocks

    @classmethod
    def from_file(cls, file_path: Path) -> Generator:
        with file_path.open(mode="r") as f:
            blocks = []
            current_block_lines = []
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#"):
                    continue
                if not line:
                    # And empty line indicates the end of the previous block
                    if current_block_lines:
                        blocks.append(Block.from_lines(current_block_lines))
                    current_block_lines = []
                if line:
                    current_block_lines.append(line)
            else:
                # Last block
                if current_block_lines:
                    blocks.append(Block.from_lines(current_block_lines))
            return cls(blocks)

    def generate_prompt(self) -> str:
        keywords = []
        activated_blocks = set()
        exclusive_blocks = set()

        for block in self.blocks:
            block_keywords = block.generate_keywords(activated_blocks, exclusive_blocks)
            if block_keywords:
                activated_blocks.add(block.name)
                exclusive_blocks.update(block.parameters.get("exclusive", "").split(","))
            keywords.extend(block_keywords)

        for block_name in activated_blocks:
            paired_blocks = [b for b in self.blocks if b.name in block.parameters.get("pair", "").split(",")]
            for paired_block in paired_blocks:
                keywords.extend(paired_block.generate_keywords(activated_blocks, exclusive_blocks))

        return ", ".join(keywords)


if __name__ == "__main__":
    _parser = ArgumentParser(
        "Creates Stable Diffusion prompts given a configuration file."
    )
    _parser.add_argument("content_file", type=Path, help="File containing the prompt parameters.")
    #_parser.add_argument("-m", "--mode", choices=("random", "exhaustive"), default="random", help="Mode of generation. Warning: exhaustive generates ALL possibilities, thus the number of prompts scales exponentially.")
    _parser.add_argument("-n", "--number", type=int, default="random", help="Number of prompts to generate. Ignored if using `mode=exhaustive`.")
    _parser.add_argument("-o", "--output", type=Path, default=Path("./"), help="File to save the prompts to.")

    args = _parser.parse_args()
    
    generator = Generator.from_file(args.content_file)
    with args.output.open("w") as f:
        for _ in range(args.number):
            f.write(f"{generator.generate_prompt()}\n")
