# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import xml.etree.ElementTree as ET
import statistics

import argparse


def paths_pruning(svg_file_path, output_file_path, opacity_delta=0.2):
    try:
        # Parse the SVG file
        tree = ET.parse(svg_file_path)
        namespace = "http://www.w3.org/2000/svg"
        ET.register_namespace("", namespace)

        root = tree.getroot()
        root.set('version', '1.1')

        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        # Collect stroke-opacity attribute values
        opacity_values = []
        for path in paths:
            opacity = path.get("stroke-opacity")
            if opacity is not None:
                opacity_values.append(float(opacity))

        # Calculate median opacity
        median_opacity = statistics.median(opacity_values) + opacity_delta

        # Create a temporary list to store paths to be removed
        paths_to_remove = []
        for path in paths:
            opacity = path.get('stroke-opacity')
            if opacity is not None and float(opacity) < median_opacity:
                paths_to_remove.append(path)

        # Remove paths from the root element
        for path in paths_to_remove:
            path.set('stroke-opacity', '0')

        print(f"n_path: {len(paths)}, "
              f"opacity_thresh: {median_opacity}, "
              f"n_path_to_remove: {len(set(paths_to_remove))}.")

        # Save the modified SVG to the specified path
        tree.write(output_file_path, encoding='utf-8', xml_declaration=True, default_namespace="")
        # print("SVG file saved successfully.")
        # print(f"file has been saved in: {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    """
    python process_svg.py -save ./xx.svg -tar ./xx.svg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-tar", "--target_file",
                        default="", type=str,
                        help="the path of SVG file place.")
    parser.add_argument("-save", "--save_path",
                        default="", type=str,
                        help="the path of processed SVG file place.")
    parser.add_argument("-od", "--opacity_delta",
                        default=0.1, type=float)
    args = parser.parse_args()

    paths_pruning(args.target_file, args.save_path, float(args.opacity_delta))
