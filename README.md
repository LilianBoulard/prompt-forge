# Stable Diffusion prompt generator

This relatively simple script aims to automate prompt generation for Stable Diffusion.

It is best used in conjonction with [AUTOMATIC111's Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) custom script "prompt file".

## Usage

The usage is quite straightforward as soon as you have you configuration file ready:

- Setup Python >= 3.11
- Run `python sd-prompt-gen.py -h` for a list of parameters you can pass to the script. The first one will always be the configuration file, like so: `python sd-prompt-gen.py my-config.toml`.
- Once you have your prompts ready in a file, go to your SD Webui, to the `txt2img` tab, and down in the scripts section, select `prompts from file`.
- Tune the other parameters (e.g., image size, negative embeddings, batch count/size, etc.).
- Click `Generate`, and wait for it to finish!

## The configuration file

To start generating prompts automatically, the first thing you need is a configuration file.
This document, written in [the `toml` format](https://toml.io/), specifies the keywords and parameters for your prompts.

### Basic usage

Blocks are the foundation of the configuration system: each block defines a set of candidate keywords that can be selected for the prompt.
The order of blocks within the file has an importance, as the picked keywords will be joined together one after the other.
Here's a simple block:

```
# Here, "my-first-block" is an arbitrary name
[blocks.my-first-block]
candidates = [
    ...
]
```

Notice our block named `my-first-block` is defined within the namespace `blocks`. This is mandatory!
Let's continue with a concrete example of the system in action. Given the following config (saved in `config.toml`):

```
[blocks.scene]
candidates = [
    "dancing folks",
    "inhabitants having lunch",
]

[blocks.place]
candidates = ["on a plaza"]
```

Running the script with this command: `python sd-prompt-gen.py config.toml --mode exhaustive` will return these prompts:

```
dancing folks, on a plaza
inhabitants having lunch, on a plaza
```

**That's it for the basics!**

### Advanced usage

A great deal of effort has also been put into providing the most sensible defaults, so that configuration files can be as simple as possible.

Despite its apparent simplicity, the system is highly customizable and allows for great complexity.

#### Candidates syntax

The candidate system provides an intuitive syntax to simplify writing keywords:
- Multiple possibilities can be specified using a pipe (` | `). For a specific part of the candidate, it has to be specified within square brackets. For instance, `[large | small] car` will result either in `large car` or `small car` (1/2 chance each). For a whole keyword, no brackets can be specified, e.g. `small van | large car` will result in either `small van` or `large car`. 
- Parts between parenthesis have a chance of not being picked. For example, `(large) car` is intuitively equivalent to `[large | ] car`, and will thus result in either `large car` or `car` (1/2 chance each). If used with a pipe for multiple options, it can replace the square brackets: for instance, `(large | small) car` will result in either `large car`, `small car` or `car` (1/3 chance each).

Both of those can be nested without restriction (e.g., `(1970 | 1980 | 1990) [((pristine) red) Ford Mustang | (sports) Porsche 911]`).

#### Blocks parameters

Blocks support a number of parameters to customize their behavior:
- `force` - This boolean parameter indicates that a keyword extracted from each candidate in the block will be included in the prompt. Incompatible with other parameters.
- `num` - This parameter takes either a positive number (e.g., `num=2`) or a range of two positive numbers (e.g., `num=1-3`). It indicates the number of candidates that will be randomly selected within the block to generate one keyword each. If unspecified, the default value is `1` (only one keyword will be selected in the block).
- `separator` - Takes a string, which will be used as the separator between entries chosen within the block (when `num`>1). By default, `", "`.
- `optional` - Boolean parameter, when `true`, shorthand for `num=0-1`, meaning: choose one keyword in the block, or none.
- `weighting` - See section `Weighting` below.

#### Groups and exlusions

For better organizational control, two systems are available:

The exclusions are groups of blocks or other groups that are incompatible, and thus cannot produce keywords for the same prompt.
They are defined within the namespace `exclusions`.

Groups, on the other hand, are sets of blocks "bundled" together. They are especially useful when combined with exclusion groups. Groups are defined within the namespace `groups`.

For example:

```
[blocks.clothing-shoes]
candidates = ["heels", "moccasin"]

[blocks.clothing-top]
candidates = ["simple shirt", "vest"]

[blocks.clothing-bottom]
candidates = ["trousers", "jeans"]

[blocks.clothing-ensemble]
candidates = ["jumpsuit", "dress"]

# Ensembles are incompatible with top and bottom clothes:
# you don't usually wear a jeans with a jumpsuit or a dress!

# Here, "clothing" is an arbitrary name
[groups.clothing]
members = ["blocks.clothing-bottom", "blocks.clothing-top"]

# So is "clothing" here
[exclusions.clothing]
members = ["blocks.clothing-ensemble", "groups.clothing"]
```

With this setup in place, it is not possible to get the prompt `heels, vest, jeans, dress`.

Also, note that their position within the configuration file has no impact.

#### Weighting

To customize the probabilistic weighting of keywords:

##### Blocks

Blocks support the parameter `weighting`, for which there are three possible values:
- `candidate-shallow` - The initial probability is spread accross the candidates, then between the options ; with each new nesting layer, the probability space is spread (thus the probability is lower and lower with each one).
- `candidate-deep` (default) - The initial probability is speach accross the candidates, then spread accross all possibilities.
- `keyword` - The probability is spread accross all keywords, so each one has an equal chance of being picked.

For example, with the configuration block:
```
[blocks.example]
weighting = "..."
candidates = ["[[large | small] | beautiful] car", "van"]
```

Here is the probability distribution for each candidate keyword depending on the weighting:

| Weighting | "*van*" | "*large car*" | "*small car*" | "*beautiful car*" |
| --- | --- | --- | --- | --- |
| `candidate-shallow` | 1/2 = 50% |  1/2 * 1/2 * 1/2 = 12.5% | 1/2 * 1/2 * 1/2 = 12.5% | 1/2 * 1/2 = 25% |
| `candidate-deep` | 1/2 = 50% | 1/2 * 1/3 = 16.6% | 1/2 * 1/3 = 16.6% | 1/2 * 1/3 = 16.6% |
| `keyword` | 1/4 = 25% | 1/4 = 25% | 1/4 = 25% | 1/4 = 25% |

###### Exclusion groups

Exclusion groups support the parameter `weights`, to which can be passed a list of integers with a size equal to the `members`.
The higher the number (weight), the higher the probability of the block/group being chosen. Example:

```
[exclusions.example]
members = [blocks.my-block, groups.my-group]
weights = [1, 2]
# `groups.my-group` will be chosen twice as much as `blocks.my-block`.
# The numbers don't really matter, the ratio between them does.
```
