# Configuration syntax

## Blocks

Block are the foundation of the configuration system: each block defines a set of keywords that can be selected for the prompt.
The order of blocks within the file has an importance, as the keywords will be joined together one after the other.

Blocks are defined with a name between square brackets. The name is purely indicative, but you should avoid special characters in them to avoid crashes. Officially supported ones are the space, the hyphen and the underscore.
After those brackets, on the same line, some parameters can be added to customize the behavior of the block.

Supported parameters are the following:
- `force` - This named parameter indicates that all keywords in the block will be included in the prompt. Incompatible with all other parameters.
- `num` - This parameter key needs a value, either a positive number (e.g., `num=2`) or a range of positive numbers (`num=1-3`). It indicates the number of keywords that will be randomly selected within the block. If not specified, the default value is 1 (only one keyword will be selected in the block).
- `optional` - Named parameter equivalent to `num=0-1`: choose one keyword in the block, or none.
- `exclusive` - This parameter takes as value a comma-separated list of block names, and indicates that the keywords in block A are incompatible with those in block B. Currently, this is a one-way parameter, meaning you should specify `exclusive=B` in block A and `exclusive=A` in block B.
- `pair` - This parameter takes as value a comma-separated list of block names, and couples the blocks together if the block is "activated" (there are candidate keywords (even if `num=0` or `optional`)). Especially useful when used with `exclusive`.

Example:
```
[Category-name](parameters)
keyword1
keyword2
...
```

Notes:
- The empty line breaks should be exclusively between blocks. Having some elsewhere will result in errors.

## Keywords

Keywords constitute a prompt. In a block, keywords are defined one per line.

There are a few syntax rules to consider when defining the keyword space:
- Multiple possibilities can be specified using a pipe (` | `). For only a part, it has to be specified within square brackets, and they can be nested. For instance, `[large | small] car` will result either in `large car` or `small car` (1/2 chance both). When we want to choose between a whole keyword or the other, no brackets can be specified, e.g. `small van | large car` will result in either `small van` or `large car`. The probability of choosing a keyword (or none) is calculated on the same layer (in case of a nested definition), and equivalent between all candidates. For example, in `[[large | small] | beautiful] car`, both `large` and `small` have a 0.5*0.5=0.25=25% chance of being picked, while `beautiful` has a 50% chance.
- Parts between parenthesis have a chance of not being picked. For example, `(large | small) car` is intuitively equivalent to `[large | small | ] car`. See bullet point above for information about choice probability.

Notes:
- Nothing prevents using commas within the keyword. This is especially useful when the keywords are block-`force`d.
- Spacing is unimportant: keywords are trimmed to get the most readable prompt.
- Using a single optional keyword argument between parenthesis in a block with `num=1`, e.g.:
  ```
  [Category]
  (keyword)
  ```
  is equivalent to using
  ```
  [Category](optional)
  keyword
  ```

## Other

You can add comments to your configuration by starting the line with "#".
