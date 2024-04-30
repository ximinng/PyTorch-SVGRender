# -*- coding: utf-8 -*-
# Author: ximing
# Description: test_svgrender
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
import argparse
import subprocess


def test_PyTorchSVGRender(args):
    # save
    result_path = f" result_path={args.result}"

    # cmd list
    run_diffvg = [
        "python svg_render.py x=diffvg target='./data/fallingwater.png' x.num_paths=512 x.num_iter=2000",
        "python svg_render.py x=diffvg target='./data/fallingwater.png' x.num_paths=1024 x.num_iter=2000"
    ]
    run_live = [
        "python svg_render.py x=live target='./data/simile.png' x.num_paths=5 x.schedule_each=1"
    ]
    run_clipasso = [
        "python svg_render.py x=clipasso target='./data/horse.png'" + " x.num_paths=8",
        "python svg_render.py x=clipasso target='./data/horse.png'" + " x.num_paths=16",
        "python svg_render.py x=clipasso target='./data/horse.png'" + " x.num_paths=24"
    ]
    run_clipascene = [
        "python svg_render.py x=clipascene target='./data/ballerina.png'",
        "python svg_render.py x=clipascene target='./data/bull.png'",
        "python svg_render.py x=clipascene target='./data/house.png'",
    ]
    run_clipdraw = [
        "python svg_render.py x=clipdraw " + "prompt='a photo of a cat'" + " seed=42",
        "python svg_render.py x=clipdraw " + "prompt='Horse eating a cupcake'",
        "python svg_render.py x=clipdraw " + "prompt='A 3D rendering of a temple'",
        "python svg_render.py x=clipdraw " + "prompt='Family vacation to Walt Disney World'",
        "python svg_render.py x=clipdraw " + "prompt='Self'",
    ]
    run_style_clipdraw = [
        "python svg_render.py x=styleclipdraw " + "prompt='A 3D rendering of a temple'" + " target='./data/starry.png'",
        "python svg_render.py x=styleclipdraw " + "prompt='Family vacation to Walt Disney World'" + " target='./data/starry.png'",
        "python svg_render.py x=styleclipdraw " + "prompt='Self'" + " target='./data/starry.png'"
    ]
    run_clipfont = [
        "python svg_render.py x=clipfont " + "prompt='Starry Night by Vincent van gogh'" + " target='./data/alphabet1.svg'",
        "python svg_render.py x=clipfont " + "prompt='English alphabet, cyberpunk'" + " target='./data/alphabet1.svg'",
        "python svg_render.py x=clipfont " + "prompt='Starry Night by Vincent van gogh'" + " target='./data/ch1.svg'",
    ]
    run_svgdreamer = [
        "python svg_render.py x=svgdreamer " + "prompt='A colorful German shepherd in vector art. tending on artstation.'" + " x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='Sydney Opera House. oil painting. by Van Gogh.'" + " x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='Seascape. Ship on the high seas. Storm. High waves. Colored ink by Mikhail Garmash. Louis Jover. Victor Cheleg.'" + " x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='Darth vader with lightsaber. ultrarealistic.'" + " x.style='pixelart' x.grid=30 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='A free-hand drawing of A speeding Lamborghini. black and white drawing.'" + " x.style = 'sketch' x.num_paths=128 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='A picture of a bald eagle. low-ploy. polygon'" + " x.style='low-poly' x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='A picture of a scarlet macaw. low-ploy. polygon'" + " x.style='low-poly' x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='a phoenix coming out of the fire drawing. lineal color. trending on artstation.'" + " x.style='painting' x.num_paths=384 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='self portrait of Van Gogh. oil painting. cmyk portrait. multi colored. defiant and beautiful. cmyk. expressive eyes.'" + " x.style='painting x.num_paths=1500 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 save_step=50",
        "python svg_render.py x=svgdreamer " + "prompt='Big Wild Goose Pagoda. ink style. Minimalist abstract art grayscale watercolor.'" + " x.style='ink' x.num_paths=128 x.width=6 x.guidance.n_particle=4 x.guidance.vsd_n_particle=2 x.guidance.phi_n_particle=2 x.guidance.t_schedule='max_0.5_2000' save_step=50"
    ]
    run_vectorfusion = [
        "python svg_render.py x=vectorfusion x.style='iconography' " + "prompt='a panda rowing a boat in a pond. minimal flat 2d vector icon. lineal color. trending on artstation.'"
    ]
    run_diffsketcher = [
        "python svg_render.py x=diffsketcher " + "prompt='a photo of Sydney opera house'" + " x.token_ind=5 seed=8019",
        "python svg_render.py x=diffsketcher " + "prompt='a photo of Sydney opera house'" + " x.token_ind=5 x.optim_width=True x.optim_rgba=True x.optim_opacity=False seed=8019"
    ]
    run_wordasimg = [
        "python svg_render.py x=wordasimage x.word='BUNNY' prompt='BUNNY' x.optim_letter='Y'",
        "python svg_render.py x=wordasimage x.word='PANTS' prompt='PANTS' x.optim_letter='P'",
        "python svg_render.py x=wordasimage x.word='FROG' prompt='FROG' x.optim_letter='G'",
        "python svg_render.py x=wordasimage x.word='DRAGONFLY' prompt='Dragonfly' x.optim_letter='Y' x.font='LuckiestGuy-Regular'"
    ]

    # test_sequence = [
    #     run_diffvg, run_live, run_clipasso, run_clipascene, run_clipdraw, run_style_clipdraw,
    #     run_clipfont, run_vectorfusion, run_diffsketcher, run_wordasimg, run_svgdreamer
    # ]
    test_dict = {
        'diffvg': run_diffvg,
        'clipdraw': run_clipdraw,
        'styleclipdraw': run_style_clipdraw,
        'live': run_live,
        'clipasso': run_clipasso,
        'clipasene': run_clipascene,
        'clipfont': run_clipfont,
        'vectorfusion': run_vectorfusion,
        'diffsketcher': run_diffsketcher,
        'wordasimg': run_wordasimg,
        'svgdreamer': run_svgdreamer
    }

    # test
    seed_str = "" if args.seed is None else f"{args.seed}"
    if args.which == 'all':  # test all scripts
        print(f"=> Testing All...")
        for test_list in test_dict.values():
            for test_cmd in test_list:
                subprocess.call(['bash', '-c', test_cmd + result_path + seed_str])
    elif args.which == 'quick':  # quick testing: test only one script per method
        print(f"=> Quick Testing...")
        for test_list in test_dict.values():
            subprocess.call(['bash', '-c', test_list[0] + result_path + seed_str])
    else:  # test all scripts for a method
        print(f"=> Testing: {args.which}... \n")
        for test_cmd in test_dict[f"{args.which}"]:
            subprocess.call(['bash', '-c', test_cmd + result_path + seed_str])

    print("all tests passed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, default='all', help='test method.')
    parser.add_argument("--result", type=str, default='./test_PyTorchSVGRender/', help='saving path.')
    parser.add_argument("--seed", type=int, default=None, help='random seed')
    args = parser.parse_args()

    """
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py
    
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'quick' --result "./quick_test_PyTorchSVGRender/" 
    
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'clipdraw' --result "./test_PyTorchSVGRender/test_clipdraw"
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'styleclipdraw' --result "./test_PyTorchSVGRender/test_styleclipdraw"
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'clipasso' --result "./test_PyTorchSVGRender/test_clipasso"
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'clipfont' --result "./test_PyTorchSVGRender/test_clipfont"
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'svgdreamer' --result "./test_PyTorchSVGRender/test_svgdreamer/"
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'wordasimg' --result "./test_PyTorchSVGRender/test_wordasimg/"
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'clipasene' --result "./test_PyTorchSVGRender/test_clipasene/"
    CUDA_VISIBLE_DEVICES=0 python test/test_svgrender.py --which 'diffsketcher' --result "./test_PyTorchSVGRender/test_diffsketcher/"
    """

    test_PyTorchSVGRender(args)
