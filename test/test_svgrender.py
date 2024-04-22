# -*- coding: utf-8 -*-
# Author: ximing
# Description: test_svgrender
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
import subprocess


def test_PyTorchSVGRender():
    # save
    result_path = " result_path='./test_PyTorchSVGRender/'"

    # cmd list
    run_diffvg = [
        "python svg_render.py x=diffvg target='./data/fallingwater.png' x.num_paths=512 x.num_iter=2000"
    ]
    run_live = [
        "python svg_render.py x=live target='./data/simile.png' x.num_paths=5 x.schedule_each=1"
    ]
    run_clipasso = [
        "python svg_render.py x=clipasso target='./data/horse.png'"
    ]
    run_clipascene = [
        "python svg_render.py x=clipascene target='ballerina.png'"
    ]
    run_clipdraw = [
        "python svg_render.py x=clipdraw " + "prompt='a photo of a cat'"
    ]
    run_style_clipdraw = [
        "python svg_render.py x=styleclipdraw " + "prompt='a photo of a cat'" + "target='./data/starry.png'"
    ]
    run_clipfont = [
        "python svg_render.py x=clipfont " + "prompt='Starry Night by Vincent van gogh'" + " target='./data/alphabet1.svg'"
    ]
    run_svgdreamer = [
        "python svg_render.py x=svgdreamer " + "prompt='A colorful German shepherd in vector art. tending on artstation.'" + " save_step=30 x.guidance.n_particle=6 x.guidance.vsd_n_particle=4 x.guidance.phi_n_particle=2"
    ]
    run_vectorfusion = [
        "python svg_render.py x=vectorfusion x.style='iconography' " + "prompt='a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. trending on artstation.'"
    ]
    run_diffsketcher = [
        "python svg_render.py x=diffsketcher " + "prompt='a photo of Sydney opera house'" + " x.token_ind=5 seed=8019",
        "python svg_render.py x=diffsketcher " + "prompt='a photo of Sydney opera house'" + " x.token_ind=5 x.optim_width=True x.optim_rgba=True x.optim_opacity=False seed=8019"
    ]
    run_wordasimg = [
        "python svg_render.py x=wordasimage x.word='BUNNY' prompt='BUNNY' x.optim_letter='Y'"
    ]

    test_sequence = [
        run_diffvg, run_live, run_clipasso, run_clipascene, run_clipdraw, run_style_clipdraw, run_style_clipdraw,
        run_clipfont, run_vectorfusion, run_diffsketcher, run_wordasimg, run_svgdreamer
    ]

    # test
    for test_list in test_sequence:
        for test_cmd in test_list:
            subprocess.call(['bash', '-c', test_cmd + result_path])

    print("all tests passed")


if __name__ == '__main__':
    test_PyTorchSVGRender()
