"""
prompt-forge
Interface for AUTOMATIC1111's Stable Diffusion webui.

Author: Lilian Boulard <https://github.com/LilianBoulard>

Licensed under the GNU Affero General Public License.
"""

import copy
import random

import gradio as gr
import modules.scripts as scripts
from modules import errors
from modules.processing import (Processed, StableDiffusionProcessing,
                                process_images)
from modules.shared import state
from prompt_forge import Generator

from scripts.prompts_from_file import cmdargs


class Script(scripts.Script):

    """
    Class for interfacing with AUTOMATIC1111's webui.
    """
    
    def title(self) -> str:
        return "Prompt forge"

    def show(self, is_img2img: bool) -> bool:
        return True

    def ui(self, is_img2img: bool) -> tuple[gr.Textbox, gr.Radio, gr.Number, gr.Checkbox, gr.Checkbox]:

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

        GUIDE_LINK = r"https://github.com/LilianBoulard/prompt-forge/blob/main/README.md"
        gr.HTML(value=f"<br>Confused/new? Read <a style=\"border-bottom: 1px #00ffff dotted;\" href=\"{GUIDE_LINK}\" target=\"_blank\" rel=\"noopener noreferrer\">the guide</a> for usage instructions!<br><br>")
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
        with gr.Row():
            generation_type = gr.Radio(
                label="Mode",
                info=(
                    "Random will create a number of prompts randomly picked from the configuration, "
                    "exhaustive will generate all possible combinations."
                ),
                choices=["Random", "Exhaustive"],
                value="Random",
            )
            number_of_entries = gr.Number(
                label="Number of random prompts to generate",
                info="Note: ignore if using exhaustive mode",
                precision=0,
                minimum=0,
            )
        with gr.Row():
            warn_if_duplicates = gr.Checkbox(
                label="Warn when there are duplicate prompts (Note: can only happen when using mode 'Random')",
                value=True,
            )
            dry_run = gr.Checkbox(
                label="Dry run to test the configuration",
                value=False,
            )

        # Push configuration file's content into the textbox
        configuration_file.change(fn=load_config_file, inputs=[configuration_file], outputs=[configuration_file, configuration, configuration], show_progress=False)
        # TODO: Listen to changes on the mode selector, and enable/disable the entries
        # selector depending on the value
        # TODO: Disable duplicates warning when using mode exhaustive

        return (
            configuration,
            generation_type,
            number_of_entries,
            warn_if_duplicates,
            dry_run,
        )

    def run(
            self,
            p: StableDiffusionProcessing,
            configuration: str,
            generation_type: str,
            number_of_entries: int,
            duplicates_warning: bool,
            dry_run: bool,
        ):
        """
        Parts of this function are from script
        [prompts_from_file](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/prompts_from_file.py)
        (https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/scripts/prompts_from_file.py).
        """
        if not configuration:
            raise RuntimeError("Configuration is required, please load one in. Read the script's guide for guidance.")

        generator: Generator = Generator.from_string(configuration)

        if generation_type == "Random":
            lines = generator.generate_random_prompts(number_of_entries)
        elif generation_type == "Exhaustive":
            lines = generator.generate_exhaustive_prompts()
        else:
            raise ValueError(f"Invalid generation type {generation_type!r}")

        if duplicates_warning:
            n_uniques = len(set(lines))
            duplicates = len(lines) - n_uniques
            if duplicates > 0:
                errors.report(f"The generated prompts contain duplicates ({duplicates}/{len(lines)})")

        if dry_run:
            print(f"Prompt-forge dry run: generated {len(lines)} prompts:")
            if len(lines) <= 8:
                for line in lines:
                    print(line)
            else:
                for line in lines[:3]:
                    print(line)
                print(f"... ({len(lines) - 6} prompts)")
                for line in lines[-3:]:
                    print(line)
            raise RuntimeWarning("Dry run completed without errors. Check the logs for details.")

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    errors.report(f"Error parsing prompt {line} as commandline", exc_info=True)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            job_count += args.get("n_iter", p.n_iter)

            jobs.append(args)

        print(f"Will process {len(lines)} prompt in {job_count} jobs.")
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