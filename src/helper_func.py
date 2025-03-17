import numpy as np
from astropy.nddata import Cutout2D
import astropy.units as u
import warnings
from dust_extinction.parameter_averages import CCM89


def color_ext_ccm89_av(wave1, wave2, av, r_v=3.1):

    model_ccm89 = CCM89(Rv=r_v)
    reddening1 = model_ccm89(wave1*u.micron) * r_v
    reddening2 = model_ccm89(wave2*u.micron) * r_v

    wave_v = 5388.55 * 1e-4
    reddening_v = model_ccm89(wave_v*u.micron) * r_v

    return (reddening1 - reddening2)*av/reddening_v


def get_img_cutout(img, wcs, coord, cutout_size):
    """function to cut out a region of a larger image with an WCS.
    Parameters
    ----------
    img : ndarray
        (Ny, Nx) image
    wcs : astropy.wcs.WCS()
        astropy world coordinate system object describing the parameter image
    coord : astropy.coordinates.SkyCoord
        astropy coordinate object to point to the selected area which to cutout
    cutout_size : float or tuple
        Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.

    Returns
    -------
    cutout : astropy.nddata.Cutout2D object
        cutout object of the initial image
    """
    if isinstance(cutout_size, tuple):
        size = cutout_size * u.arcsec
    elif isinstance(cutout_size, float) | isinstance(cutout_size, int):
        size = (cutout_size, cutout_size) * u.arcsec
    else:
        raise KeyError('cutout_size must be float or tuple')

    # check if cutout is inside the image
    pix_pos = wcs.world_to_pixel(coord)
    if (pix_pos[0] > 0) & (pix_pos[0] < img.shape[1]) & (pix_pos[1] > 0) & (pix_pos[1] < img.shape[0]):
        return Cutout2D(data=img, position=coord, size=size, wcs=wcs)
    else:
        warnings.warn("The selected cutout is outside the original dataset. The data and WCS will be None",
                      DeprecationWarning)
        cut_out = type('', (), {})()
        cut_out.data = None
        cut_out.wcs = None
        return cut_out


def plot_reddening_vect(ax,
                        x_wave_1=3366, x_wave_2=4339,  y_wave_1=5389, y_wave_2=8117,
                        init_x_color=0, init_y_color=0, av_value_mag=1,
                        arrow_line_width=2, arrow_line_color='k',
                        show_av_value=False, text_font_size=20, text_color='k', x_text_offset=0.1, y_text_offset=-0.3):
    """
    Function to plot reddening vector on color-color diagram based on reddening law described in
    Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model

    Parameters
    ----------
    ax : ``matplotlib.pylab.Axes``
        axis on which to plot
    x_wave_1, x_wave_2, y_wave_1, y_wave_2 : int or float
        Wavelengths in Angstrom for the 4 used colors. The colors are calculated as x_wave_1 - x_wave_2 and
        y_wave_1 - y_wave_2 on the X- and Y-axes, respectively.
    init_x_color , init_y_color : float
        initial x and Y color from where the reddening vector stars
    av_value_mag : float
        A_v value to determining the length of the reddening vector

    arrow_line_width : float or int
    arrow_line_color : str
    show_av_value : bool
        flag to show the A_v value as a text
    text_font_size : float or int
    text_color : str
    x_text_offset , y_text_offset : float
        in units of x and y colors the amount you need to offset the text from the arrow to avoid overlapping


    Returns
    -------
    None

    """
    # change wavelength unit from Angstrom into micron
    x_wave_1 *= 1e-4
    x_wave_2 *= 1e-4
    y_wave_1 *= 1e-4
    y_wave_2 *= 1e-4
    # calculate the color difference from reddening
    color_ext_x = color_ext_ccm89_av(wave1=x_wave_1, wave2=x_wave_2, av=av_value_mag)
    color_ext_y = color_ext_ccm89_av(wave1=y_wave_1, wave2=y_wave_2, av=av_value_mag)
    # plot arrow annotation with the calculated color differences
    ax.annotate('', xy=(init_x_color + color_ext_x, init_y_color + color_ext_y), xycoords='data',
                xytext=(init_x_color, init_y_color), fontsize=text_font_size,
                textcoords='data', arrowprops=dict(arrowstyle='-|>', color=arrow_line_color, lw=arrow_line_width, ls='-'))
    # add A_v value as text
    if show_av_value:
        if isinstance(av_value_mag, int):
            arrow_text = r'A$_{\rm V}$=%i mag' % av_value_mag
        else:
            arrow_text = r'A$_{\rm V}$=%.1f mag' % av_value_mag
        # calculate the angle of the reddening vector in order to annotate text parallel to it
        angle_av_vector = np.arctan(color_ext_y/color_ext_x) * 180/np.pi
        ax.text(init_x_color + x_text_offset, init_y_color + y_text_offset, arrow_text,
                horizontalalignment='left', verticalalignment='bottom',
                transform_rotates_text=True, rotation_mode='anchor',
                rotation=angle_av_vector, fontsize=text_font_size, color=text_color)
