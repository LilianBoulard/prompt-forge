"""
prompt-forge

Author: Lilian Boulard <https://github.com/LilianBoulard>

Licensed under the GNU Affero General Public License.
"""

from __future__ import annotations

import copy
import json
import random
import re
import tomllib
from itertools import product
from pathlib import Path
from typing import Literal

import gradio as gr
import jsonschema
import modules.scripts as scripts
from modules import errors
from modules.processing import Processed, process_images
from modules.shared import state
from processing import StableDiffusionProcessing
from prompts_from_file import cmdargs


def is_balanced(parens: str) -> bool:
    # From: https://stackoverflow.com/a/73341167/
    parens_map ={"(":")","{":"}","[":"]"}
    stack = []
    for paren in parens:
        if paren in parens_map:  # is open
            stack.append(paren)
        elif paren in parens_map.values():  # is closed
            if (not stack) or (paren != parens_map[stack.pop()]):
                return False
    return not stack


def build_candidate_tree(candidate_part: list | str) -> Candidate:
    if isinstance(candidate_part, list):
        node = Candidate()  # Create dummy
        # Add children
        for item in candidate_part:
            node.children.append(build_candidate_tree(item))
    else:
        # Create leaf
        node = Candidate(candidate_part)
    return node


def blocks_in_group(element: Block | Group | ExclusionGroup) -> set[Block]:
    """
    Given an element, lists the blocks it is constituted of.

    Parameters
    ----------
    element: Block, Group or ExclusionGroup
        Element to flatten.

    Returns
    -------
    set of Block
        The element, flattened to only the blocks it contains.
    """
    blocks = []
    members_queue = [element]
    while members_queue:
        member = members_queue.pop()
        if isinstance(member, Block):
            blocks.append(member)
        elif isinstance(member, (Group, ExclusionGroup)):
            members_queue.extend(member.members)
    return set(blocks)


def groups_in_group(group: Group | ExclusionGroup) -> list[Group | ExclusionGroup]:
    groups = []
    members_queue = group.members.copy()
    while members_queue:
        member = members_queue.pop()
        if isinstance(member, Block):
            continue
        elif isinstance(member, (Group, ExclusionGroup)):
            groups.append(member)
            members_queue.extend(member.members)
    return set(groups)


class Candidate:

    """
    A candidate node (used in a tree) either has a value
    (it's a leaf) or a list of children (it's a dummy).
    """

    def __init__(self, keyword: str | None = None):
        self.value = keyword
        self.children: list[Candidate] = []
    
    def __repr__(self) -> list:
        if self.value is not None:
            return str(self.value)
        return f"{[repr(child) for child in self.children]}"

    def expand(self) -> str | list[str]:
        """
        Expands the tree to enumerate all keywords.

        Returns
        -------
        str
            If the candidate is a leaf, returns its value.
        list[str]
            The list of all keywords.
        """
        if (
            (self.value is not None and self.children)
            or (self.value is None and not self.children)
        ):
            errors.report(
                "Candidate must either be a leaf "
                "(have a value) or a dummy (has children). " 
                f"Has both: {self.value=}, {self.children=}"
            )

        # If it's a leaf, return its value
        if self.value is not None:
            return self.value

        # First pass: we append everything to a single list.
        # e.g. ["a", "b", ["c", "d"], "e"]
        prompt_parts: list[str] = [
            child.expand()
            for child in self.children
        ]

        # Second pass: we put continuously single values
        # in lists.
        # We leave the already existing lists alone.
        # e.g. [["a", "b"], ["c", "d"], ["e"]]
        temp_prompts: list[str] = []
        buffer = []
        for prompt_part in prompt_parts:
            if isinstance(prompt_part, str):
                buffer.append(prompt_part)
            elif isinstance(prompt_part, list):
                # Bundle previous values together
                if buffer:
                    temp_prompts.append(buffer.copy())
                    buffer.clear()
                # Store this one
                temp_prompts.append(prompt_part)
        if buffer:
            temp_prompts.append(buffer.copy())

        # Finally, we do the final product
        # e.g. ["a c e", "a d e", "b c e", "b d e"]       
        return [
            " ".join(pair).strip()
            for pair in product(*temp_prompts)
        ]


class Group:

    def __init__(self, name: str, members: list[Block]):
        self.name = name
        self.members = members

    def __hash__(self):
        return hash(self.name)
    
    def __repr__(self):
        return f"Group {self.name!r} with {len(self.members)} members"


class ExclusionGroup:

    def __init__(self, name: str, members: list[Block | Group], weights: list[int] | None):
        self.name = name
        self.members = members
        self.weights = weights

    def choose_member(self) -> Block | Group:
        return random.choices(self.members, weights=self.weights)[0]

    def __hash__(self):
        return hash(self.name)
    
    def __repr__(self):
        return f"ExclusionGroup {self.name!r} with {len(self.members)} members"


class Block:

    """
    A block of keywords.

    Parameters
    ----------
    name: str
        Name of the block.
    parameters: mapping of str to str or bool
        Parameters passed in the config.
    candidates: list of str
        Candidates (can contain square brackets and parentheses).

    Attributes
    ----------
    name: str
        Name of the block
    num: range
        Possible number of keywords to pick.
    separator: str
        When picking multiple keywords from this block,
        the separator to use when joining them.
    force: bool
        Whether all keywords in this block should be included in the prompt.
        Note: not all combinations, all keywords.
    keywords: list of str
        Exhaustive list of all keywords that can be picked from this block.
    weights: list of int or None
        Generated at init, specifies the weight of each keyword.
        Can be None, in which case the weight is equal for each keyword.
    weighting: {"candidate-shallow", "candidate-deep", "keyword"}
        How to tune probabilities when picking a keyword.
        Refer to the guide, section "Weighting" for more information.
    """

    name: str
    num: range
    separator: str
    force: bool
    keywords: list[str]
    weights: list[int] | None
    weighting: Literal["candidate-shallow", "candidate-deep", "keyword"]

    def __init__(self, name: str, candidates: list[str], parameters: dict[str, any]) -> Block:
        self.name = name
        self.update_parameters(parameters)
        self.keywords, self.weights = self._generate_keywords(candidates)

    def __hash__(self):
        return hash(self.name)
    
    def __repr__(self):
        return f"Block {self.name!r} with {len(self.keywords)} keywords"

    def update_parameters(self, parameters: dict[str, any]) -> None:
        """
        In-place method that takes a dictionary of parameters
        and overrides the current values of the instance.

        Parameters
        ----------
        parameters: dict of str to any value
            The new parameters for the instance.
            Unsupported parameters are ignored by this function.
        """
        self.num = range(1, 2)
        if num_param := parameters.get("num"):
            # Parses `2` as `(2, 2)` and `2-3` as `(2, 3)`
            min_num, max_num = map(int, num_param.split("-")) if "-" in num_param else (int(num_param), int(num_param))
            self.num = range(min_num, max_num + 1)
        if parameters.get("optional"):
            self.num = range(0, 2)

        self.weighting = parameters.get("weighting", "candidate-deep")
        assert self.weighting in {"candidate-shallow", "candidate-deep", "keyword"}

        self.separator = parameters.get("separator", ", ")

        self.force = parameters.get("force", False)

    def _generate_keywords(self, keyword_candidates: list[str]) -> tuple[list[str], list[int] | None]:
        """
        Takes keyword candidates and expands them all,
        returning an exhaustive list of all keywords
        that can be picked from the block, and their weights.

        Parameters
        ----------
        keyword_candidates: list of str
            Keyword candidates to expand

        Returns
        -------
        2-tuple of a list of str and a list of int
            The first list is the exhaustive list of keywords.
            The second one contains the weights for the choice of a parameter.
        """
        # Preliminary pass:
        # 1. Clean up
        # 2. Check validity
        # 3. Convert to candidate tree
        candidates: list[Candidate] = []
        for candidate in keyword_candidates:
            candidate = candidate.replace("(", "[").replace(")", " | ]")
            # Avoids unexpected prompts
            if not is_balanced(candidate):
                errors.report(f"Unbalanced brackets in {candidate!r}")
            candidate_repr = self._construct_parts(candidate)
            candidate_tree = build_candidate_tree(candidate_repr)
            candidates.append(candidate_tree)

        if self.weighting == "candidate-shallow":
            # TODO
            errors.report("`candidate-shallow` is not yet implemented")
            raise NotImplemented

        elif self.weighting == "candidate-deep":
            final_keywords: list[tuple[str, int]] = []
            # First step: create the exhaustive list for the candidate
            # Second step: divide the probas by the number of candidates
            groups: dict[int, list[str]] = {}
            for group, candidate in enumerate(candidates):
                groups.update({group: candidate.expand()})
            # Scale by the total number of entries
            total_entries = sum(map(len, groups.values()))
            for group, entries in groups.items():
                weight = round(1 / len(entries) * total_entries)
                final_keywords.extend([
                    (entry, weight)
                    for entry in entries
                ])
            return list(zip(*final_keywords))  # unzip

        elif self.weighting == "keyword":
            # Create the exhaustive list for the candidate,
            # give an equal weight for each.
            final_keywords_nested: list[list[str]] = [
                candidate.expand()
                for candidate in candidates
            ]
            # Flatten
            final_keywords: list[str] = [
                keyword
                for nested_keywords in final_keywords_nested
                for keyword in nested_keywords
            ]
            return final_keywords, None

        else:
            # FIXME: should be handled by the schema validation
            errors.report(
                f"Expected parameter weighting to be any of "
                f"{{'candidate-shallow', 'candidate-deep', 'keyword'}}, "
                f"got {self.weighting!r}. "
            )

    def generate_keyword(self) -> str:
        return self.separator.join(
            # FIXME: can choose multiple keywords from a same candidate,
            # which is probably not the expected behavior
            random.choices(
                self.keywords,
                self.weights,
                k=min(random.choice(self.num), len(self.keywords)),
            )
        )

    def _construct_parts(self, candidate: str) -> list:
        """
        Takes the string representation of a candidate,
        and converts it to a tree-style structure, which
        can be further processed.

        Parameters
        ----------
        candidate: str
            The candidate keyword.
            E.g. "[[pristine | ] red] Ford Mustang | [sports | ] Porsche 911"

        Returns
        -------
        nested list of str
            The tree-style representation of the candidate.
            E.g. [[[[['pristine', ''], 'red']], 'Ford Mustang'], [['sports', ''], 'Porsche 911']]
        """
        # Since we strip a lot, we will explicitly add spaces around
        # the pipes, otherwise the splitting method use down below
        # might not work (it wouldn't find the empty string).
        # E.g. `an| example|` -> `an | example | `.
        candidate = re.sub(r"(\s{0,1})\|(\s{0,1})", lambda x: f" {x.group(0).strip()} ", candidate)

        # First pass: split based on the pipes
        split_parts = []
        buffer = []
        nesting_level = 0
        for c in candidate:
            if c == "[":
                nesting_level += 1
            elif c == "]":
                nesting_level -= 1

            buffer.append(c)

            if c == "|" and nesting_level == 0:
                split_parts.append("".join(buffer[:-1]).strip())
                buffer.clear()

        if buffer:
            split_parts.append("".join(buffer).strip())
            buffer.clear()

        # Second pass: isolate nested values
        isolated_parts: list[str | list[str]] = []
        buffer = []
        nesting_level = 0
        for part in split_parts:
            nested_parts_buffer: list[str] = []
            if part == "":
                isolated_parts.append(part)
                continue
            if "[" in part:
                # There are nested values.
                # If there is a value besides (e.g. `car` in `[sports | ] car`),
                # then we must keep them together (they form a single group)
                # so we'll nest them.
                for c in part:
                    if c == "[":
                        nesting_level += 1
                        if nesting_level == 1:
                            if buffer:
                                nested_parts_buffer.append("".join(buffer).strip())
                                buffer.clear()
                            continue
                    elif c == "]":
                        nesting_level -= 1
                        if nesting_level == 0:
                            if buffer:
                                nested_parts_buffer.append("".join(buffer).strip())
                                buffer.clear()
                            continue
                    buffer.append(c)
                if buffer:
                    nested_parts_buffer.append("".join(buffer).strip())
                    buffer.clear()
                isolated_parts.append(nested_parts_buffer)
            else:
                # The part does not have a nested value
                isolated_parts.append(part)

        # Third pass: call recursively and collect the results
        parsed_parts = []
        for part in isolated_parts:
            if isinstance(part, str):
                if "|" in part:
                    parsed_parts.append(self._construct_parts(part))
                else:
                    parsed_parts.append(part)
            elif isinstance(part, list):
                nested_part_parts = []
                for nested_part in part:
                    if "|" in nested_part or "[" in nested_part:
                        nested_part_parts.append(self._construct_parts(nested_part))
                    else:
                        nested_part_parts.append(nested_part)
                parsed_parts.append(nested_part_parts)

        return parsed_parts


class Generator:

    """
    A generator created from a config.
    Capable of creating new prompts on demand.
    """

    def __init__(self, elements: list[Block | Group | ExclusionGroup], blocks_order: dict[str, int]) -> Generator:
        self.elements = elements
        self.blocks_order = blocks_order

    @classmethod
    def from_string(cls, configuration: str) -> Generator:
        # Parse the config
        config = tomllib.loads(configuration)

        # Since toml returns the config as an unordered JSON document,
        # we read the configuration to find the order of the blocks
        pattern = re.compile(r"\[blocks\.(.+?)\]")
        blocks_order: dict[str, int] = {
            match.group(0): match.start()
            for match in pattern.finditer(configuration)
        }

        # Validate the JSON schema
        schema_file = Path(__file__).parent / "config-schema.json"
        if schema_file.is_file():
            with schema_file.open("rb") as f:
                schema = json.load(f)
            jsonschema.validate(config, schema)
        else:
            errors.report(
                f"Did not find schema at {schema_file!s} "
                f"to validate the configuration against. "
            )

        # Create the blocks
        blocks = {
            Block(name, block.get("candidates", list()), block)
            for name, block in config["blocks"].items()
        }
        mappings = {f"blocks.{block.name}": block for block in blocks}

        # Create the groups
        groups = [
            Group(
                name,
                [
                    mappings[group_name]
                    for group_name in group.get("members", list())
                ],
            )
            for name, group in config["groups"].items()
        ]
        mappings.update({f"groups.{group.name}": group for group in groups})

        # Create the exclusion groups
        exclusion_groups = [
            ExclusionGroup(
                name, 
                [
                    mappings[member_name]
                    for member_name in group.get("members", list())
                ],
                group.get("weights"),
            )
            for name, group in config["exclusions"].items()
        ]

        # List blocks that are present in at least one group
        used_blocks = {
            block
            for group in {*groups, *exclusion_groups}
            for block in blocks_in_group(group)
        }
        # List groups that are present in exclusion groups
        groups_in_exclusion_groups = {
            group
            for exclusion_group in exclusion_groups
            for group in groups_in_group(exclusion_group)
        }

        # List the blocks that are not present in any groups
        elements = blocks.difference(used_blocks)
        # List groups present in exclusion groups
        leftover_groups = {
            group
            for group in groups
            if group not in groups_in_exclusion_groups
        }

        # Add the remaining groups
        elements.update(leftover_groups)
        # And the exclusion groups
        elements.update(exclusion_groups)

        return cls(elements, blocks_order)

    def sort_elements(self, element: Block | Group | ExclusionGroup) -> int:
        return min(
            self.blocks_order[f"[blocks.{block.name}]"]
            for block in blocks_in_group(element)
        )

    def generate_random_prompts(self, n: int) -> list[str]:
        prompts = []
        for _ in range(n):
            prompt = []
            activated_blocks = []
            stack = [*self.elements]
            while stack:
                element = stack.pop()
                if isinstance(element, Block):
                    activated_blocks.append(element)
                elif isinstance(element, Group):
                    stack.extend(element.members)
                elif isinstance(element, ExclusionGroup):
                    stack.append(element.choose_member())
            for block in sorted(activated_blocks, key=self.sort_elements):
                keyword = block.generate_keyword()
                if keyword:  # Ignore empty keywords
                    prompt.append(keyword)
            prompts.append(", ".join(prompt))
        return prompts

    def generate_exhaustive_prompts(self) -> list[str]:
        # TODO
        raise NotImplemented


class Script(scripts.Script):

    """
    Class for interfacing with AUTOMATIC1111's webui.
    """
    
    def title(self) -> str:
        return "Prompt forge"

    def show(self, is_txt2img: bool, is_img2img: bool) -> bool:
        return is_txt2img or is_img2img

    def ui(self) -> tuple[gr.Textbox, gr.Radio, gr.Number]:

        def load_config_file(file: gr.File):
            """
            From https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/scripts/prompts_from_file.py#L96
            FIXME: why 7?
            """
            if file is None:
                return None, gr.update(), gr.update(lines=7)
            else:
                lines = file.decode('utf8', errors='ignore').splitlines()
                return None, "\n".join(lines), gr.update(lines=7)

        def switch_number_selector(mode_selector: gr.Radio, number_selector: gr.Number) -> None:
            if mode_selector.value == "Random":
                number_selector.visible = True
            elif mode_selector.value == "Exhaustive":
                number_selector.visible = False
            else:
                errors.report(f"Got unexpected generation mode {mode_selector.value!r}")

        configuration = gr.Textbox(
            label="Configuration",

            elem_id=self.elem_id("prompt_txt"),
        )
        configuration_file = gr.File(
            label="Configuration file",
            info="See project's README for syntax definition",
            file_types=[".toml"],
            type="binary",
            elem_id=self.elem_id("configuration_file"),
        )
        generation_type = gr.Radio(
            label="Mode",
            info=(
                "Random will create a number of prompts randomly picked from the configuration, "
                "exhaustive will generate all possible combinations."
            ),
            choices=["Random", "Exhaustive"],
        )
        number_of_entries = gr.Number(
            label="Number of random prompts to generate",
            precision=0,
            minimum=0,
        )
        warn_if_duplicates = gr.Checkbox(
            label="Warn when there are duplicate prompts",
            info="Note: can only happen when using mode 'Random'",
            value=True,
        )

        # Push configuration file's content into the textbox
        configuration_file.change(fn=load_config_file, inputs=[configuration_file], outputs=[configuration_file, configuration, configuration], show_progress=False)
        # Listen to changes on the mode selector, and enable/disable the entries
        # selector depending on the value
        generation_type.change(switch_number_selector, inputs=[generation_type, number_of_entries])

        return configuration, generation_type, number_of_entries, warn_if_duplicates

    def run(
            self,
            p: StableDiffusionProcessing,
            configuration: gr.Textbox,
            generation_type: gr.Radio,
            number_of_entries: gr.Number,
            duplicates_warning: gr.Checkbox,
        ):
        """
        Parts of this function are from script
        [prompts_from_file](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/prompts_from_file.py)
        (https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/scripts/prompts_from_file.py).
        """
        generator: Generator = Generator.from_string(configuration)

        if generation_type == "Random":
            lines = generator.generate_random_prompts(number_of_entries)
        elif generation_type == "Exhaustive":
            lines = generator.generate_exhaustive_prompts()
        else:
            errors.report(f"Invalid generation type {generation_type!r}")

        if duplicates_warning:
            n_uniques = len(set(lines))
            duplicates = len(lines) - n_uniques
            if duplicates > 0:
                errors.report(f"The generated prompts contain duplicates ({duplicates}/{len(lines)})")

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    errors.report(f"Error parsing line {line} as commandline", exc_info=True)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            job_count += args.get("n_iter", p.n_iter)

            jobs.append(args)

        print(f"Will process {len(lines)} lines in {job_count} jobs.")
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images

            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
