import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import TextIO

from scripts.prompt_forge import Generator


if __name__ == "__main__":
    parser = ArgumentParser(
        "Prompt-forge: create Stable Diffusion prompts from a configuration file"
    )

    parser.add_argument("content_file", type=Path, help="File containing the prompt generation configuration.")
    parser.add_argument("-m", "--mode", choices=("random", "exhaustive"), default="random", help="Mode of generation. Warning: exhaustive generates ALL possibilities, thus the number of prompts scales exponentially.")  # TODO
    parser.add_argument("-n", "--number", type=int, required=False, help="Number of prompts to generate. Ignored if `mode=exhaustive`.")
    parser.add_argument("-o", "--output", type=Path, default=sys.stdout, help="File to save the prompts to. By default, outputs to stdout.")

    args = parser.parse_args()

    if args.mode == "random" and not args.number:
        raise ValueError("Expected a number of prompts to be generated with `mode=random`")

    generator = Generator.from_file(args.content_file)

    def write(prompts: list[str], destination: Path | TextIO):
        prompts_txt = "\n".join(prompts)
        if isinstance(destination, Path):
            destination.write_text(prompts_txt)
        else:
            destination.write(prompts_txt)

    if args.mode == "random":
        write(generator.generate_random_prompts(n=args.number), args.output)
    elif args.mode == "exhaustive":
        write(generator.generate_exhaustive_prompts(), args.output)
