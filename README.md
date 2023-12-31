# Stable Diffusion prompt forge

This script aims to automate prompt generation for Stable Diffusion (and more generally, txt2img models such as MidJourney, Dall-E, etc.).

It is an extension designed for [AUTOMATIC1111's Stable Diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui),
but is also available as a standalone script.

If you like the project, :star: it on Github, and share it to your SD friends!

<hr>

- [Stable Diffusion prompt forge](#stable-diffusion-prompt-forge)
  - [Install](#install)
    - [AUTOMATIC1111 stable diffusion webui](#automatic1111-stable-diffusion-webui)
    - [Standalone](#standalone)
  - [Demo](#demo)
  - [The configuration file](#the-configuration-file)
    - [Basic usage](#basic-usage)
    - [Advanced usage](#advanced-usage)
      - [Candidates syntax](#candidates-syntax)
      - [Block types](#block-types)
      - [Blocks parameters](#blocks-parameters)
      - [Groups and exclusions](#groups-and-exclusions)
      - [Weighting](#weighting)
        - [Blocks](#blocks)
        - [Exclusion groups](#exclusion-groups)
  - [Similar projects](#similar-projects)

<hr>

## Install

### AUTOMATIC1111 stable diffusion webui

Like for any other AUTOMATIC1111 webui extension, go to the `Extension` tab, then into `Install from URL`, and paste this repo's URL (https://github.com/LilianBoulard/prompt-forge). Finally, click `Install`, reload the UI and you're ready to roll!

### Standalone

A standalone script `prompt-forge-standalone.py` is provided so the system can be run via a command-line interface.
To use it, you'll need to install Python and the following packages:
- `toml` if using Python<=10
- `jsonschema`

Once that's done, run `python prompt-forge-standalone.py -h` for a description of the options.

<hr>

## Demo

[![prompt-forge demo, video available on YouTube](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DSJBXHkra5Go)](https://www.youtube.com/watch?v=SJBXHkra5Go)

<hr>

## The configuration file

To start forging prompts, you need a **configuration file**: this document, written in [the `toml` format](https://toml.io/), specifies the keywords and parameters for your prompts.

### Basic usage

**Blocks** are the foundation of the configuration system: each block defines a set of candidate keywords that can be selected for the prompt.
The order of blocks within the file has an importance: they will have the same order in the prompt.
Here's a simple block:

```toml
# Here, "my-first-block" is an arbitrary name
[blocks.prompt.my-first-block]
candidates = [
    ...
]
```

Notice our block is located in the namespace `blocks`. `prompt` means the candidates will contribute to the prompt. `my-first-block` is the name of the prompt block.
Let's continue with a concrete example of the system in action. Given the following config (saved in `config.toml`):

```toml
[blocks.prompt.scene]
candidates = [
    "dancing folks",
    "people having lunch",
]

[blocks.prompt.place]
candidates = ["on a plaza"]
```

Running the script with in mode `Exhaustive` will return these prompts:

```
dancing folks, on a plaza
inhabitants having lunch, on a plaza
```

**That's it for the basics!**

<hr>

### Advanced usage

A great deal of effort has also been put into providing the most sensible defaults, so that configuration files can be as short as possible.

Despite its apparent simplicity, the system is highly customizable and allows for great complexity.

#### Candidates syntax

The **candidate system** provides an intuitive way of simplifying writing keywords:
- Multiple possibilities can be specified using a pipe (` | `). For a specific part of the candidate, it has to be specified within square brackets. For instance, `[large | small] car` will result either in `large car` or `small car` (1/2 chance each). For a whole keyword, no brackets can be specified, e.g. `small van | large car` will result in either `small van` or `large car`. 
- Parts between parenthesis have a chance of not being picked. For example, `(large) car` is intuitively equivalent to `[large | ] car`, and will thus result in either `large car` or `car` (1/2 chance each). If used with a pipe for multiple options, it can replace the square brackets: for instance, `(large | small) car` will result in either `large car`, `small car` or `car` (1/3 chance each).

Both of those can be nested without restriction (e.g., `(1970 | 1980 | 1990) [((pristine) red) Ford Mustang | (sports) Porsche 911]`).

#### Block types

A wide range of parameters can be set via the configuration file. Blocks are defined as `blocks.<block-type>.<block-name>`. Here's an exhaustive list of supported values:

- `outpath_samples` - string
- `outpath_grids` - string
- `prompt_for_display` - string
- `prompt` - string
- `negative_prompt` - string
- `styles` - string
- `seed` - int
- `subseed_strength` - float
- `subseed` - int
- `seed_resize_from_h` - int
- `seed_resize_from_w` - int
- `sampler_index` - int
- `sampler_name` - string
- `batch_size` - int
- `n_iter` - int
- `steps` - int
- `cfg_scale` - float
- `width` - int
- `height` - int
- `restore_faces` - boolean
- `tiling` - boolean
- `do_not_save_samples` - boolean
- `do_not_save_grid` - boolean

Note: the type indicated refers to the candidates' type within the block. Although they should have the format of their type, values should be specified as strings.
For example:

- `int`: `["1", "2", "55"]`
- `float`: `["6.2", "0.1", "9.4"]`
- `boolean`: `["true", "false"]`

#### Blocks parameters

Blocks support the following parameters for customizing their behavior:
- `force` - This boolean parameter indicates that a keyword extracted from each candidate in the block will be included in the prompt. Shorthand of `num=<number of candidates>`.
- `num` - This parameter takes either a positive number (e.g., `num=2`) or a range of two positive numbers (e.g., `num=1-3`). It indicates the number of candidates that will be randomly selected within the block to generate one keyword each. If unspecified, the default value is `1` (only one candidate will be selected in the block to generate a keyword).
- `separator` - Takes a string, which will be used as the separator between entries chosen within the block (when `num`>1). By default, `", "`.
- `optional` - Boolean parameter, when `true`, shorthand for `num=0-1`, meaning: choose one keyword in the block, or none.
- `weighting` - See section `Weighting` below.

#### Groups and exclusions

For better organizational control, two systems are available:

The exclusions are groups of blocks or other groups that are incompatible, and thus cannot produce keywords for the same prompt.
They are defined within the namespace `exclusions`.

Groups, on the other hand, are sets of blocks "bundled" together. They are especially useful when combined with exclusion groups. Groups are defined within the namespace `groups`.

For example:

```toml
[blocks.prompt.clothing-shoes]
candidates = ["heels", "moccasin"]

[blocks.prompt.clothing-top]
candidates = ["simple shirt", "vest"]

[blocks.prompt.clothing-bottom]
candidates = ["trousers", "jeans"]

[blocks.prompt.clothing-ensemble]
candidates = ["jumpsuit", "dress"]

# Ensembles are incompatible with top and bottom clothes:
# you don't usually wear a jeans with a jumpsuit or a dress!

# Here, "clothing" is an arbitrary name
[groups.clothing]
members = ["blocks.prompt.clothing-bottom", "blocks.prompt.clothing-top"]

# So is "clothing" here
[exclusions.clothing]
members = ["blocks.prompt.clothing-ensemble", "groups.clothing"]
```

With this setup in place, it is not possible to get the prompt `heels, vest, jeans, dress`.

As with all blocks not defined within the namespace `blocks`, their position within the configuration file is unimportant.

#### Weighting

To customize the probabilistic weighting of keywords:

##### Blocks

Blocks support the parameter `weighting`, for which there are three possible values:
- `candidate-shallow` - The initial probability is spread accross the candidates, then between the options ; with each new nesting layer, the probability is spread (thus is lower and lower with each one).
- `candidate-deep` (default) - The initial probability is speach accross the candidates, then spread accross all possibilities.
- `keyword` - The probability is spread accross all keywords, so each one has an equal chance of being picked.

For example, with this configuration block:

```toml
[blocks.prompt.example]
weighting = "..."
candidates = [
    "[[large | small] | beautiful] car",
    "van",
]
```

here's the probability distribution depending on the weighting:

| | `candidate-shallow` | `candidate-deep` | `keyword` |
| --- | --- | --- | --- |
| **"*van*"** | 1/2 = 50% | 1/2 = 50% | 1/4 = 25% |
| **"*large car*"** | 1/2 * 1/2 * 1/2 = 12.5% | 1/2 * 1/3 = 16.6% | 1/4 = 25% |
| **"*small car*"** | 1/2 * 1/2 * 1/2 = 12.5% | 1/2 * 1/3 = 16.6% | 1/4 = 25% |
| **"*beautiful car*"** | 1/2 * 1/2 = 25% | 1/2 * 1/3 = 16.6% | 1/4 = 25% |

##### Exclusion groups

Exclusion groups support the parameter `weights`, which takes a list of integers. Its size must be equal to the size of `members`.
The higher the number (weight), the higher the probability of the block/group being chosen. For example:

```toml
[exclusions.example]
members = ["blocks.my-block", "groups.my-group"]
weights = [1, 2]
# `groups.my-group` will be chosen twice as much as `blocks.my-block`.
# The numbers don't really matter, only the ratio does.
```

## Similar projects

| Project | its pros | prompt-forge pros |
|---|---|---|
| [sd-dynamic-prompts](https://github.com/adieyal/sd-dynamic-prompts) | - Some creative features, out-of-scope of prompt-forge<br>  (i.e. magic prompt, attention grabber, I'm feeling lucky)<br>- More complex syntax<br>- More mature and with a larger community | - Designed for industrial-scale generation<br>- Though for people that already know what they want<br>- Simpler and more descriptive syntax<br>- Easier to version with git or alike since it's a single conf file |
