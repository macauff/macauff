# The MIT License (MIT)

# Copyright (c) 2015 timothydmorton

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# Licensed under a 3-clause BSD style license - see LICENSE
'''
Provides the code to query the online TRILEGAL API and download the results.
'''


import os
import re
import subprocess as sp
import time

import dustmaps.sfd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import UnitsError

__all__ = []


def get_trilegal(filename, ra, dec, folder='.', galactic=False,
                 filterset='kepler_2mass', area=1, magnum=1, maglim=27, binaries=False,
                 trilegal_version='1.7', av=None, sigma_av=0.1):
    """
    Calls the TRILEGAL web form simulation and downloads the file.

    Parameters
    ----------
    filename : string
        Output filename. If extension not provided, it will be added.
    ra : float
        Coordinate for line-of-sight simulation.
    dec : float
        Coordinate for line-of-sight simulation.
    folder : string, optional
        Folder to which to save file.
    filterset : string, optional
        Filter set for which to call TRILEGAL.
    area : float, optional
        Area of TRILEGAL simulation [sq. deg]
    magnum : integer, optional
        Bandpass number to limit source magnitudes in.
    maglim : integer, optional
        Limiting magnitude in ``magnum`` bandpass of the ``filterset``.
    binaries : boolean, optional
        Whether to have TRILEGAL include binary stars. Default ``False``.
    trilegal_version : float, optional
        Version of the TRILEGAL API to call. Default ``'1.7'``.
    av : float, optional
        Extinction at infinity in the V band. If not provided, defaults to
        ``None``, in which case it is calculated from the coordinates given.
    sigma_av : float, optional
        Fractional spread in A_V along the line of sight.

    Returns
    -------
    av : float
        The extinction at infinity of the particular TRILEGAL call, if
        ``av`` as passed into the function is ``None``, otherwise just
        returns the input value.
    result : string
        ``timeout`` or ``good`` depending on whether the run was successful
        or not.
    """
    frame = 'galactic' if galactic else 'icrs'
    try:
        c = SkyCoord(ra, dec, frame=frame)
    except UnitsError:
        c = SkyCoord(ra, dec, unit='deg', frame=frame)
    l, b = (c.galactic.l.value, c.galactic.b.value)

    if os.path.isabs(filename):
        folder = ''
        outfolder = os.path.dirname(filename)
    else:
        outfolder = folder

    if not re.search(r'\.dat$', filename):
        outfile = f'{folder}/{filename}.dat'
    else:
        outfile = f'{folder}/{filename}'
    if av is None:
        av = get_av_infinity(l, b, frame='galactic')[0]

    result = trilegal_webcall(trilegal_version, l, b, area, binaries, av, sigma_av, filterset,
                              magnum, maglim, outfile, outfolder)

    return av, result


def trilegal_webcall(trilegal_version, l, b, area, binaries, av, sigma_av, filterset, magnum,
                     maglim, outfile, outfolder):
    """
    Calls TRILEGAL webserver and downloads results file.

    Parameters
    ----------
    trilegal_version : float
        Version of TRILEGAL.
    l : float
        Galactic coordinate for line-of-sight simulation.
    b : float
        Galactic coordinate for line-of-sight simulation.
    area : float
        Area of TRILEGAL simulation in square degrees.
    binaries : boolean
        Whether to have TRILEGAL include binary stars.
    av : float
        Extinction along the line of sight.
    sigma_av : float
        Fractional spread in A_V along the line of sight.
    filterset : string
        Filter set for which to call TRILEGAL.
    magnum : integer
        Number of filter in given filterset to limit magnitudes to.
    maglim : float
        Limiting magnitude down to which to simulate sources.
    outfile : string
        Output filename.
    outfolder : string
        Output filename's containing folder.

    Returns
    -------
    string
        ``timeout`` or ``good`` depending on whether the run was successful
        or not.
    """
    webserver = 'http://stev.oapd.inaf.it'
    mainparams = ('imf_file=tab_imf%2Fimf_chabrier_lognormal.dat&binary_frac=0.3&'
                  'binary_mrinf=0.7&binary_mrsup=1&extinction_h_r=100000&extinction_h_z='
                  f'110&extinction_kind=2&extinction_rho_sun=0.00015&extinction_infty={av}&'
                  f'extinction_sigma={sigma_av}&r_sun=8700&z_sun=24.2&thindisk_h_r=2800&'
                  'thindisk_r_min=0&thindisk_r_max=15000&thindisk_kind=3&thindisk_h_z0='
                  '95&thindisk_hz_tau0=4400000000&thindisk_hz_alpha=1.6666&'
                  'thindisk_rho_sun=59&thindisk_file=tab_sfr%2Ffile_sfr_thindisk_mod.dat&'
                  'thindisk_a=0.8&thindisk_b=0&thickdisk_kind=0&thickdisk_h_r=2800&'
                  'thickdisk_r_min=0&thickdisk_r_max=15000&thickdisk_h_z=800&'
                  'thickdisk_rho_sun=0.0015&thickdisk_file=tab_sfr%2Ffile_sfr_thickdisk.dat&'
                  'thickdisk_a=1&thickdisk_b=0&halo_kind=2&halo_r_eff=2800&halo_q=0.65&'
                  'halo_rho_sun=0.00015&halo_file=tab_sfr%2Ffile_sfr_halo.dat&halo_a=1&'
                  'halo_b=0&bulge_kind=2&bulge_am=2500&bulge_a0=95&bulge_eta=0.68&'
                  'bulge_csi=0.31&bulge_phi0=15&bulge_rho_central=406.0&'
                  'bulge_cutoffmass=0.01&bulge_file=tab_sfr%2Ffile_sfr_bulge_zoccali_p03.dat&'
                  'bulge_a=1&bulge_b=-2.0e9&object_kind=0&object_mass=1280&object_dist=1658&'
                  'object_av=1.504&object_avkind=1&object_cutoffmass=0.8&'
                  'object_file=tab_sfr%2Ffile_sfr_m4.dat&object_a=1&object_b=0&'
                  'output_kind=1')
    cmd = (f"wget -o {outfolder}/lixo -O {outfolder}/tmpfile --post-data='submit_form=Submit&"
           f"trilegal_version={trilegal_version}&gal_coord=1&gc_l={l}&gc_b={b}&eq_alpha=0&eq_delta=0&"
           f"field={area}&photsys_file=tab_mag_odfnew%2Ftab_mag_{filterset}.dat&icm_lim={magnum}&"
           f"mag_lim={maglim}&mag_res=0.1&binary_kind={binaries}&{mainparams}' "
           f"{webserver}/cgi-bin/trilegal_{trilegal_version}")
    complete = False
    while not complete:  # pylint: disable=too-many-nested-blocks
        notconnected = True
        busy = True
        print("TRILEGAL is being called with \n l={l} deg, b={b} deg, area={area} sqrdeg\n "
              "Av={av} with {sigma_av} fractional r.m.s. spread \n in the {filterset} system, complete "
              f"down to mag={maglim} in its {magnum}th filter, use_binaries set to {binaries}.")
        sp.Popen(cmd, shell=True).wait()  # pylint: disable=consider-using-with
        if (os.path.exists(f'{outfolder}/tmpfile') and
                os.path.getsize(f'{outfolder}/tmpfile') > 0):
            notconnected = False
        else:
            print(f"No communication with {webserver}, will retry in 2 min")
            time.sleep(120)
            return "nocomm"
        if not notconnected:
            with open(f'{outfolder}/tmpfile', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                if 'The results will be available after about 2 minutes' in line:
                    busy = False
                    save_line = line
                    break
            # pylint: disable-next=consider-using-with
            sp.Popen(f'rm -f {outfolder}/lixo {outfolder}/tmpfile', shell=True)
            if not busy:
                filenameidx = save_line.find('<a href=../tmp/') + 15
                fileendidx = save_line[filenameidx:].find('.dat')
                filename = save_line[filenameidx:filenameidx+fileendidx+4]
                print(f"retrieving data from {filename} ...")
                while not complete:
                    time.sleep(40)
                    modcmd = f'wget -o {outfolder}/lixo -O {outfolder}/{filename} {webserver}/tmp/{filename}'
                    sp.Popen(modcmd, shell=True).wait()  # pylint: disable=consider-using-with
                    if os.path.getsize(f'{outfolder}/{filename}') > 0:
                        with open(f'{outfolder}/{filename}', 'r', encoding='utf-8') as f:
                            lastline = f.readlines()[-1]
                        if 'normally' in lastline:
                            complete = True
                            print('model downloaded!..')
                    if not complete:
                        print('still running...')
            else:
                print('Server busy, trying again in 2 minutes')
                time.sleep(120)
                # The way the "breakout" return calls work now we don't loop
                # within trilegal_webcall any more, but the loops and if
                # statements are left in for backwards compatibility.
                return "timeout"
    sp.Popen(f'mv {outfolder}/{filename} {outfile}', shell=True).wait()  # pylint: disable=consider-using-with
    print(f'results copied to {outfile}')

    return "good"


def get_av_infinity(ra, dec, frame='icrs'):
    """
    Gets the Schlegel, Finkbeiner & Davis 1998 (ApJ, 500, 525) A_V extinction
    at infinity for a given line of sight, using the updated parameters from
    Schlafly & Finkbeiner 2011 (ApJ 737, 103), table 6.

    Parameters
    ----------
    ra : float, list, or numpy.ndarray
        Sky coordinate.
    dec : float, list, or numpy.ndarray
        Sky coordinate.
    frame : string, optional
        Frame of input coordinates (e.g., ``'icrs', 'galactic'``)

    Returns
    -------
    av : numpy.ndarray
        Extinction at infinity as given by the SFD dust maps for the chosen sky
        coordinates.
    """
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    coords = SkyCoord(ra, dec, unit='deg', frame=frame).transform_to('galactic')

    sfd_ebv = dustmaps.sfd.SFDQuery()
    av = 2.742 * sfd_ebv(coords)

    return av
