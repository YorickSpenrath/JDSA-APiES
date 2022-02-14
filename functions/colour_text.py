import warnings
from typing import Iterable

from functions.general_functions import listified


def get_style_string(**kwargs):
    """
    Get style string for given arguments

    Other Parameters
    ----------------
    colour: str
        Optional. Text colour, see Notes.
    style: (Iterable of) str
        Optional. Text style, one of 'bold', 'underline', 'italic'. If iterable, each will be applied.
    background: str
        Optional. Background colour, see Notes.

    Notes
    -----
    Acceptable (background) colours are: 'k' (black), 'r' (red), 'g' (green), 'b' (blue), 'y' (yellow), 'c' (cyan),
    'm' (magenta), 'w' (white)

    If 'color' is provided bu not 'colour', that value is used for 'colour'

    Returns
    -------
    z: str
        style part of the mark up.
    """

    # To prevent future issues with simplified English
    if 'color' in kwargs and 'colour' not in kwargs:
        kwargs['colour'] = kwargs['color']
        del kwargs['color']

    foo = 'krgybpcw'
    style_string_elements = []

    # Style parameter
    if 'style' in kwargs:
        style_dict = {'bold': '1', 'underline': '4', 'italic': '3'}
        for z in listified(kwargs.pop('style'), str):
            style_string_elements += [style_dict[z]]

    # Text colour parameter
    if 'colour' in kwargs:
        colour_dict = {k: f'9{v}' for v, k in enumerate(foo)}
        style_string_elements += [colour_dict[kwargs.pop('colour')]]

    # Background colour parameter
    if 'background' in kwargs:
        bg_colour_dict = {k: f'4{v}' for v, k in enumerate(foo)}
        style_string_elements += [bg_colour_dict[kwargs.pop('background')]]

    # Any remaining kwargs cannot be interpreted
    if kwargs:
        warnings.warn(f'Cannot interpret keywords to format string {kwargs.keys()}')

    # Return
    return ';'.join(style_string_elements) + 'm'


def makeup_string(s, **kwargs):
    """
    Mark up a string.

    Parameters
    ----------
    s: str
        The string to mark up

    Other Parameters
    ----------------
    See :py:meth:`get_style_string(**kwargs)<functions.colour_text.get_style_string>` for details

    Returns
    -------
    s_mark_up: str
        String with markup

    """
    return highlight_substring(s, s, **kwargs)


def highlight_substring(s, h, **kwargs):
    """
    Mark up a substring.

    Parameters
    ----------
    s: str
        Str to be highlighted
    h: str, int, tuple, None, Iterable
        If str, highlight specific substring. If int, highlight character at position. If tuple of two ints, highlight
        substring h[0]:h[1]. If None: do not highlight. If Iterable, call function with each element.

    Other Parameters
    ----------------
    See :py:meth:`get_style_string(**kwargs)<functions.colour_text.get_style_string>` for details

    Returns
    -------
    s_hl: str
        String with highlighted options
    """

    # Styling
    style_string = get_style_string(**kwargs)

    if h is None:
        return s
    elif isinstance(h, int):
        return highlight_substring(s, (h, h + 1))
    elif isinstance(h, tuple) and len(h) == 2 and isinstance(h[0], int) and isinstance(h[1], int):
        return s[:h[0]] + f'\033[{style_string}{s[h[0]:h[1]]}\033[0m' + s[h[1]:]
    elif isinstance(h, str):
        return s.replace(h, f'\033[{style_string}{h}\033[0m')
    elif isinstance(h, Iterable):
        for hi in h:
            s = highlight_substring(s, hi)
        return s
    else:
        raise NotImplementedError


def _showcase():
    for x in 'rgbykwpc':
        z = ''
        print(makeup_string('text_only', colour=x))
        print(makeup_string('bg_only', background=x))
        for y in 'rgbykwpc':
            z += makeup_string('x', colour=x, background=y)
        print(z)

    print(makeup_string('text', style=['bold', 'underline']))
    print(makeup_string('text', style='underline'))
    print(makeup_string('text', style='italic'))

    print(makeup_string('Drago was here', style=['italic', 'underline', 'bold'], colour='y', background='k'))


if __name__ == '__main__':
    _showcase()
