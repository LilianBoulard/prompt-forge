"""
prompt-forge
Core logic code

Author: Lilian Boulard <https://github.com/LilianBoulard>

Licensed under the GNU Affero General Public License.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Literal

import jsonschema

if sys.version_info.major == 3 and sys.version_info.minor <= 10:
    import toml as tomllib
else:
    import tomllib


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


def blocks_in_element(element: Block | Group | ExclusionGroup) -> set[Block]:
    """
    Given an element, lists the blocks it is constituted of.

    Parameters
    ----------
    element : Block, Group or ExclusionGroup
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

    def __init__(self, *, keyword: str | None = None, children: list | None = None, operator: Literal["OR", "AND"] | None = None):
        self.value = keyword
        self.children = children
        self.operator = operator

    def __len__(self) -> int:
        if self.value is not None:
            return 1
        else:
            return sum(len(child) for child in self.children)

    @classmethod
    def parse(cls, candidate: str) -> Candidate:
        """
        Takes the string representation of a candidate,
        and converts it to a tree-style structure, which
        can be further processed.

        Parameters
        ----------
        candidate : str
            The candidate keyword.

        Returns
        -------
        Candidate
            The tree representation of the candidate.
        """
        if not is_balanced(candidate):
            raise ValueError(f"Candidate {candidate!r} has unbalanced brackets/parentheses")

        candidate = candidate.replace("(", "[").replace(")", " | ]")

        # We will explicitly add spaces around
        # the pipes and the brackets, otherwise the splitting method
        # used down below might not work (it wouldn't find the empty string).
        # E.g. `a| [simple| example]|` -> `a | [ simple | example] | `.
        candidate = re.sub(r"(\s{0,1})\|(\s{0,1})", lambda x: f" {x.group(0).strip()} ", candidate)
        candidate = re.sub(r"(\s{0,1})\[(\s{0,1})", lambda x: f" {x.group(0).strip()} ", candidate)
        candidate = re.sub(r"(\s{0,1})\](\s{0,1})", lambda x: f" {x.group(0).strip()} ", candidate)

        return cls._parse(candidate)

    @classmethod
    def _parse(cls, candidate: str) -> Candidate:
        # First pass: split based on the pipes
        # (only on the first nesting layer,
        # we'll delegate the nested ones to recursive calls)
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
        isolated_parts: list[list[str]] = []
        buffer = []
        nesting_level = 0
        for part in split_parts:
            nested_parts_buffer: list[str] = []
            if part == "":
                isolated_parts.append([part])
                continue
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

        # Third pass: call recursively and collect the results
        children: list[Candidate] = []
        for part in isolated_parts:
            inner_buffer: list[Candidate] = []
            for nested_part in part:
                if "|" in nested_part or "[" in nested_part:
                    inner_buffer.append(cls.parse(nested_part))
                else:
                    inner_buffer.append(cls(keyword=nested_part))
            if len(inner_buffer) == 1:
                children.append(inner_buffer[0])
            else:
                children.append(cls(children=inner_buffer, operator="AND"))

        return cls(children=children, operator="OR")

    def expand(self, *, weighting: Literal["candidate-shallow", "candidate-deep", "keyword"]) -> list[tuple[str, int]]:
        """
        Expands the tree to enumerate all keywords.

        Parameters
        ----------
        weighting : {"candidate-shallow", "candidate-deep", "keyword"}
            Weighting system to use for this candidate (inherited from the block).
            Refer to the README for more information.

        Returns
        -------
        list of 2-tuples of str and int
            All keywords that can be picked from this candidate,
            along with the individual's weight.
        """
        if (
            (self.value is not None and self.children and self.operator)
            or (self.value is None and not self.children and not self.operator)
        ):
            raise ValueError(
                f"Candidate must either be a leaf "
                f"(have a value) or a dummy (have children and an operator). " 
                f"Has an incorrect combination: {self.value=}, "
                f"{self.children=}, {self.operator=}"
            )

        # If it's a leaf, return its value with a single unit of weight
        if self.value is not None:
            return [(self.value, 1)]

        children_and_op: list[tuple[list[tuple[str, int]], Literal["OR", "AND"]]] = [
            # If it's a leaf, it won't have an operator,
            # so we default to AND
            (child.expand(weighting=weighting), child.operator or "AND")
            for child in self.children
        ]

        if self.operator == "AND":
            # Explode AND lists with multiple elements.
            # Keep OR lists nested.
            # TODO: take care of the weights
            product_candidates = []
            for keywords, operator in children_and_op:
                if operator == "AND":
                    for item, weight in keywords:
                        product_candidates.append([(item, weight)])
                else:
                    product_candidates.append(keywords)
            prod = list(product(*product_candidates))
            joined_keywords = []
            for elements in prod:
                # FIXME: weighting is incorrect
                keywords, weights = zip(*elements)
                joined_keywords.append((
                    " ".join(filter(lambda keyword: not not keyword, keywords)),
                    min(weights),
                ))
            return joined_keywords

        elif self.operator == "OR":
            # Return everything together
            if weighting == "candidate-shallow":
                # Adjust the cumulative weights
                raise NotImplementedError("`candidate-shallow` is not implemented yet")
            elif weighting == "candidate-deep" or weighting == "keyword":
                # Do not adjust the cumulative weights
                return [
                    # weight is always 1
                    (item, weight)
                    for items, _ in children_and_op
                    for item, weight in items
                ]


class Group:

    __namespace__ = "groups"

    def __init__(self, name: str, members: list[Block]) -> Group:
        self.name = name
        self.members = members

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return f"Group {self.fqn!r} <{len(self.members)} members>"

    @property
    def fqn(self) -> str:
        """Fully qualified name"""
        return f"{self.__namespace__}.{self.name}"


class ExclusionGroup:

    __namespace__ = "exclusions"

    def __init__(self, name: str, members: list[Block | Group], weights: list[int] | None) -> ExclusionGroup:
        self.name = name
        self.members = members
        self.weights = weights

    def choose_member(self) -> Block | Group:
        return random.choices(self.members, weights=self.weights)[0]

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return f"ExclusionGroup {self.fqn!r} <{len(self.members)} members>"

    @property
    def fqn(self) -> str:
        """Fully qualified name"""
        return f"{self.__namespace__}.{self.name}"


class Block:

    """
    A block of keywords.

    Parameters
    ----------
    name : str
        Name of the block.
    type : str
        Block type.
    candidates : list of str
        Candidates (can contain square brackets and parentheses).
    parameters : mapping of str to str or bool
        Parameters passed in the config.

    Attributes
    ----------
    name : str
        Name of the block
    type : str
        Block type
    num : range
        Possible number of keywords to pick.
    separator : str
        When picking multiple keywords from this block,
        the separator to use when joining them.
    candidates : list of list of 2-tuples of str and int
        The keywords for each candidate.
    weighting : {"candidate-shallow", "candidate-deep", "keyword"}
        How to tune probabilities when picking a keyword.
        Refer to the guide, section "Weighting" for more information.
    """

    __namespace__ = "blocks"

    name: str
    type: str
    num: range
    separator: str
    candidates: list[list[tuple[str, int]]]
    weighting: Literal["candidate-shallow", "candidate-deep", "keyword"]

    def __init__(self, name: str, type: str, candidates: list[str], parameters: dict[str, any]) -> Block:
        self.name = name
        self.type = type

        self.weighting = parameters.get("weighting", "candidate-deep")
        assert self.weighting in {"candidate-shallow", "candidate-deep", "keyword"}

        self.candidates = [
            Candidate.parse(candidate).expand(weighting=self.weighting)
            for candidate in candidates
        ]

        self.num = range(1, 2)
        if parameters.get("force"):
            self.num = range(len(self.candidates), len(self.candidates) + 1)
        elif parameters.get("optional"):
            self.num = range(0, 2)
        elif num_param := parameters.get("num"):
            # Parses `2` as `(2, 2)` and `2-3` as `(2, 3)`
            min_num, max_num = map(int, num_param.split("-")) if "-" in num_param else (int(num_param), int(num_param))
            self.num = range(min_num, max_num + 1)

        self.separator = parameters.get("separator", ", ")

    def __hash__(self) -> int:
        return hash(self.fqn)

    def __repr__(self) -> str:
        return (
            f"Block {self.fqn!r} "
            f"<{len(self.candidates)} candidates, "
            f"{sum(len(keywords) for keywords in self.candidates)} keywords>"
        )

    @property
    def fqn(self) -> str:
        """Fully qualified name"""
        return f"{self.__namespace__}.{self.type}.{self.name}"

    def generate_keyword(self) -> str:
        k = min(random.choice(self.num), len(self.candidates))
        chosen_candidates: list[list[tuple[str, int]]]

        if self.weighting in {"candidate-shallow", "candidate-deep"}:
            chosen_candidates = random.choices(self.candidates, k=k)
            return self.separator.join([
                random.choices(*zip(*candidate))[0]
                for candidate in chosen_candidates
            ])

        elif self.weighting == "keyword":
            candidate_weights = [len(candidate) for candidate in self.candidates]
            chosen_candidates = random.choices(self.candidates, weights=candidate_weights, k=k)

    def generate_all_keywords(self) -> list[str]:
        """
        Used when mode="exhaustive", returns the list of
        all possible keywords this block can generate.
        """
        return [
            keyword
            for candidate_keywords in self.candidates
            for keyword, _ in candidate_keywords
        ]


class Generator:

    """
    A generator created from a config.
    Capable of creating new prompts on demand.
    """

    def __init__(self, elements: list[Block | Group | ExclusionGroup], blocks_order: dict[str, int]) -> Generator:
        self.elements = elements
        self.blocks_order = blocks_order

    @classmethod
    def from_file(cls, file: Path) -> Generator:
        with file.open() as f:
            return cls.from_string(f.read())

    @classmethod
    def from_string(cls, configuration: str) -> Generator:
        # Parse the config
        config = tomllib.loads(configuration)

        # Validate the JSON schema
        schema_file = Path(__file__).parent / "config-schema.json"
        if schema_file.is_file():
            with schema_file.open("rb") as f:
                schema = json.load(f)
            jsonschema.validate(config, schema)
        else:
            print(
                f"Did not find schema at {schema_file!s} "
                f"to validate the configuration against. "
                f"There might be unexpected/weird errors "
                f"during execution. "
            )

        # Since toml returns the config as an unordered JSON document,
        # we read the configuration to find the order of the blocks
        pattern = re.compile(r"\[" + Block.__namespace__ + r"\.(.+?)\]")
        blocks_order: dict[str, int] = {
            match.group(0): match.start()
            for match in pattern.finditer(configuration)
        }

        # Create the blocks
        blocks = {
            Block(block_name, category, block_info.get("candidates", list()), block_info)
            for category, contained_blocks in config.get(Block.__namespace__, dict()).items()
            for block_name, block_info in contained_blocks.items()
        }
        mappings = {block.fqn: block for block in blocks}

        # Create the groups
        groups = [
            Group(
                name=name,
                members=[
                    mappings[group_name]
                    for group_name in group.get("members", list())
                ],
            )
            for name, group in config.get(Group.__namespace__, dict()).items()
        ]
        mappings.update({group.fqn: group for group in groups})

        # Create the exclusion groups
        exclusion_groups = [
            ExclusionGroup(
                name=name, 
                members=[
                    mappings[member_name]
                    for member_name in group.get("members", list())
                ],
                weights=group.get("weights"),
            )
            for name, group in config.get(ExclusionGroup.__namespace__, {}).items()
        ]

        # List blocks that are present in at least one group
        used_blocks = {
            block
            for group in {*groups, *exclusion_groups}
            for block in blocks_in_element(group)
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

    def _divide_by_type(self, blocks: list[Block]) -> list[tuple[str, list[Block]]]:
        """
        Takes a list of blocks, and returns them categorized by type.
        """
        divided_blocks = defaultdict(list)
        for block in blocks:
            divided_blocks[block.type].append(block)
        return list(divided_blocks.items())

    def _block_order(self, block: Block) -> int:
        """
        Returns an index that can be used for sorting blocks.
        """
        return self.blocks_order[f"[{block.fqn}]"]

    def generate_random_prompts(self, n: int) -> list[str]:
        """
        Generate `n` random prompts.

        Parameters
        ----------
        n : int
            Number of random prompts to generate.

        Returns
        -------
        list of str
            The random prompts. There can be duplicate prompts.
        """
        prompts = []
        for _ in range(n):
            prompt = []
            activated_blocks: list[Block] = []
            stack = [*self.elements]
            while stack:
                element = stack.pop()
                if isinstance(element, Block):
                    activated_blocks.append(element)
                elif isinstance(element, Group):
                    stack.extend(element.members)
                elif isinstance(element, ExclusionGroup):
                    stack.append(element.choose_member())
            for category, blocks in self._divide_by_type(activated_blocks):
                keywords = []
                for block in sorted(blocks, key=self._block_order):
                    if keyword := block.generate_keyword():  # Ignore empty keywords
                        keywords.append(keyword)
                if keywords:
                    prompt.append(f"--{category} " + ", ".join(keywords))
            prompts.append(" ".join(prompt))
        return prompts

    def generate_exhaustive_prompts(self) -> list[str]:
        """
        Generates all possible prompts.

        Returns
        -------
        list of str
            Generated prompts. There are no duplicates,
            but the list can be very long.
        """
        # First step: create all the lines, that is,
        # lists of blocks that will generate prompts
        lines: list[list[Block]] = [[]]
        for element in self.elements:
            if isinstance(element, Block):
                for line in lines:
                    line.append(element)
            elif isinstance(element, ExclusionGroup):
                new_lines = []
                for member in element.members:
                    alternative_lines = lines.copy()
                    for line in alternative_lines:
                        line.extend(blocks_in_element(member))
                    new_lines.extend(alternative_lines)
                lines = new_lines

        # Second step: create the prompts
        prompts: list[str] = []
        for line in lines:
            prompt_parts: list[list[str]] = []
            for category, blocks in self._divide_by_type(line):
                keywords: list[list[str]] = []
                for block in sorted(blocks, key=self._block_order):
                    keywords.append(block.generate_all_keywords())
                category_prompts = [
                    f"--{category} " + ", ".join(sub_prompt_parts)
                    for sub_prompt_parts in product(*keywords)
                ]
                prompt_parts.append(category_prompts)

            line_prompts: list[list[str]] = product(*prompt_parts)
            prompts.extend(map(lambda prompt_parts: " ".join(prompt_parts), line_prompts))

        return prompts
