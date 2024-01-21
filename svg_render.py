# -*- coding: utf-8 -*-
# Author: ximing xing
# Description: the main func of this project.
# Copyright (c) 2023, XiMing Xing.

import os
import sys
from functools import partial

from accelerate.utils import set_seed
import hydra
import omegaconf

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from pytorch_svgrender.utils import render_batch_wrap, get_seed_range

METHODS = ['live',
           'vectorfusion',
           'clipasso',
           'clipascene',
           'diffsketcher',
           'stylediffsketcher',
           'clipdraw',
           'styleclipdraw',
           'wordasimage',
           'clipfont',
           'svgdreamer']


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(cfg: omegaconf.DictConfig):
    # print(omegaconf.OmegaConf.to_yaml(cfg))
    flag = cfg.x.method
    assert flag in METHODS, f"{flag} is not currently supported!"

    # seed prepare
    set_seed(cfg.seed)
    seed_range = get_seed_range(cfg.srange) if cfg.multirun else None

    # render function
    render_batch_fn = partial(render_batch_wrap, cfg=cfg, seed_range=seed_range)

    if flag == "wordasimage":  # text2font
        from pytorch_svgrender.pipelines.WordAsImage_pipeline import WordAsImagePipeline

        pipe = WordAsImagePipeline(cfg)
        pipe.painterly_rendering(cfg.x.word, cfg.prompt, cfg.x.optim_letter)

    elif flag == "clipfont":  # text and font to font
        from pytorch_svgrender.pipelines.CLIPFont_pipeline import CLIPFontPipeline

        if not cfg.multirun:
            pipe = CLIPFontPipeline(cfg)
            pipe.painterly_rendering(svg_path=cfg.target, prompt=cfg.prompt)
        else:  # generate many SVG at once
            render_batch_fn(pipeline=CLIPFontPipeline, svg_path=cfg.target, prompt=cfg.prompt)

    elif flag == "live":  # img2svg
        from pytorch_svgrender.pipelines.LIVE_pipeline import LIVEPipeline

        pipe = LIVEPipeline(cfg)
        pipe.painterly_rendering(cfg.target)

    elif flag == "vectorfusion":  # text2svg
        from pytorch_svgrender.pipelines.VectorFusion_pipeline import VectorFusionPipeline

        if not cfg.multirun:
            pipe = VectorFusionPipeline(cfg)
            pipe.painterly_rendering(cfg.prompt)
        else:  # generate many SVG at once
            render_batch_fn(pipeline=VectorFusionPipeline, text_prompt=cfg.prompt)

    elif flag == "svgdreamer":  # text2svg
        from pytorch_svgrender.pipelines.SVGDreamer_pipeline import SVGDreamerPipeline

        if not cfg.multirun:
            pipe = SVGDreamerPipeline(cfg)
            pipe.painterly_rendering(cfg.prompt)
        else:  # generate many SVG at once
            render_batch_fn(pipeline=SVGDreamerPipeline, text_prompt=cfg.prompt, target_file=None)

    elif flag == "clipasso":  # img2sketch
        from pytorch_svgrender.pipelines.CLIPasso_pipeline import CLIPassoPipeline

        pipe = CLIPassoPipeline(cfg)
        pipe.painterly_rendering(cfg.target)

    elif flag == 'clipascene':
        from pytorch_svgrender.pipelines.CLIPascene_pipeline import CLIPascenePipeline

        pipe = CLIPascenePipeline(cfg)
        pipe.painterly_rendering(cfg.target)

    elif flag == "clipdraw":  # text2svg
        from pytorch_svgrender.pipelines.CLIPDraw_pipeline import CLIPDrawPipeline

        pipe = CLIPDrawPipeline(cfg)
        pipe.painterly_rendering(cfg.prompt)

    elif flag == "styleclipdraw":  # text to stylized svg
        from pytorch_svgrender.pipelines.StyleCLIPDraw_pipeline import StyleCLIPDrawPipeline

        pipe = StyleCLIPDrawPipeline(cfg)
        pipe.painterly_rendering(cfg.prompt, style_fpath=cfg.target)

    elif flag == "diffsketcher":  # text2sketch
        from pytorch_svgrender.pipelines.diffsketcher_pipeline import DiffSketcherPipeline

        if not cfg.multirun:
            pipe = DiffSketcherPipeline(cfg)
            pipe.painterly_rendering(cfg.prompt)
        else:  # generate many SVG at once
            render_batch_fn(pipeline=DiffSketcherPipeline, prompt=cfg.prompt)

    elif flag == "stylediffsketcher":  # text2sketch + style transfer
        from pytorch_svgrender.pipelines.diffsketcher_stylized_pipeline import StylizedDiffSketcherPipeline

        if not cfg.multirun:
            pipe = StylizedDiffSketcherPipeline(cfg)
            pipe.painterly_rendering(cfg.prompt, style_fpath=cfg.target)
        else:  # generate many SVG at once
            render_batch_fn(pipeline=StylizedDiffSketcherPipeline, prompt=cfg.prompt, style_fpath=cfg.style_file)


if __name__ == '__main__':
    main()
