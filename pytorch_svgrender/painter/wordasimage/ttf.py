import numpy as np
import freetype as ft

from . import bezier


def glyph_to_cubics(face, x=0):
    """Convert current font face glyph to cubic beziers"""

    def linear_to_cubic(Q):
        a, b = Q
        return [a + (b - a) * t for t in np.linspace(0, 1, 4)]

    def quadratic_to_cubic(Q):
        return [Q[0],
                Q[0] + (2 / 3) * (Q[1] - Q[0]),
                Q[2] + (2 / 3) * (Q[1] - Q[2]),
                Q[2]]

    beziers = []
    pt = lambda p: np.array([p.x + x, -p.y])  # Flipping here since freetype has y-up
    last = lambda: beziers[-1][-1]

    def move_to(a, beziers):
        beziers.append([pt(a)])

    def line_to(a, beziers):
        Q = linear_to_cubic([last(), pt(a)])
        beziers[-1] += Q[1:]

    def conic_to(a, b, beziers):
        Q = quadratic_to_cubic([last(), pt(a), pt(b)])
        beziers[-1] += Q[1:]

    def cubic_to(a, b, c, beziers):
        beziers[-1] += [pt(a), pt(b), pt(c)]

    face.glyph.outline.decompose(beziers, move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
    beziers = [np.array(C).astype(float) for C in beziers]
    return beziers


def font_string_to_beziers(font, txt, size=30, spacing=1.0, merge=True, target_control=None):
    """
    Load a font and convert the outlines for a given string to cubic bezier curves,
        if merge is True, simply return a list of all bezier curves,
        otherwise return a list of lists with the bezier curves for each glyph
    """

    face = ft.Face(font)
    face.set_char_size(64 * size)
    slot = face.glyph

    x = 0
    beziers = []
    previous = 0
    for c in txt:
        face.load_char(c, ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP)
        bez = glyph_to_cubics(face, x)

        # Check number of control points if desired
        if target_control is not None:
            if c in target_control.keys():
                nctrl = np.sum([len(C) for C in bez])
                while nctrl < target_control[c]:
                    longest = np.max(
                        sum([[bezier.approx_arc_length(b) for b in bezier.chain_to_beziers(C)] for C in bez], []))
                    thresh = longest * 0.5
                    bez = [bezier.subdivide_bezier_chain(C, thresh) for C in bez]
                    nctrl = np.sum([len(C) for C in bez])
                    print("nctrl: ", nctrl)

        if merge:
            beziers += bez
        else:
            beziers.append(bez)

        kerning = face.get_kerning(previous, c)
        x += (slot.advance.x + kerning.x) * spacing
        previous = c

    return beziers


def bezier_chain_to_commands(C, closed=True):
    curves = bezier.chain_to_beziers(C)
    cmds = 'M %f %f ' % (C[0][0], C[0][1])
    n = len(curves)
    for i, bez in enumerate(curves):
        if i == n - 1 and closed:
            cmds += 'C %f %f %f %f %f %fz ' % (*bez[1], *bez[2], *bez[3])
        else:
            cmds += 'C %f %f %f %f %f %f ' % (*bez[1], *bez[2], *bez[3])
    return cmds


def write_letter_svg(c, header, fontname, beziers, subdivision_thresh, dest_path):
    cmds = ''
    svg = header

    path = '<g><path d="'
    for C in beziers:
        if subdivision_thresh is not None:
            print('subd')
            C = bezier.subdivide_bezier_chain(C, subdivision_thresh)
        cmds += bezier_chain_to_commands(C, True)
    path += cmds + '"/>\n'
    svg += path + '</g></svg>\n'

    fname = f"{dest_path}/{fontname}_{c}.svg"
    fname = fname.replace(" ", "_")
    with open(fname, 'w') as f:
        f.write(svg)
    return fname, path
