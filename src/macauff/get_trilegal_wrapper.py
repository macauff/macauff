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
                 trilegal_version='1.7', AV=None, sigma_AV=0.1):
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
    AV : float, optional
        Extinction at infinity in the V band. If not provided, defaults to
        ``None``, in which case it is calculated from the coordinates given.
    sigma_AV : float, optional
        Fractional spread in A_V along the line of sight.

    Returns
    -------
    AV : float
        The extinction at infinity of the particular TRILEGAL call, if
        ``AV`` as passed into the function is ``None``, otherwise just
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
        outfile = '{}/{}.dat'.format(folder, filename)
    else:
        outfile = '{}/{}'.format(folder, filename)
    if AV is None:
        AV = get_AV_infinity(l, b, frame='galactic')[0]

    result = trilegal_webcall(trilegal_version, l, b, area, binaries, AV, sigma_AV, filterset,
                              magnum, maglim, outfile, outfolder)

    return AV, result


def trilegal_webcall(trilegal_version, l, b, area, binaries, AV, sigma_AV, filterset, magnum,
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
    AV : float
        Extinction along the line of sight.
    sigma_AV : float
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
    args = [l, b, area, AV, sigma_AV, filterset, maglim, magnum, binaries]
    mainparams = ('imf_file=tab_imf%2Fimf_chabrier_lognormal.dat&binary_frac=0.3&'
                  'binary_mrinf=0.7&binary_mrsup=1&extinction_h_r=100000&extinction_h_z='
                  '110&extinction_kind=2&extinction_rho_sun=0.00015&extinction_infty={}&'
                  'extinction_sigma={}&r_sun=8700&z_sun=24.2&thindisk_h_r=2800&'
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
                  'output_kind=1').format(AV, sigma_AV)
    cmdargs = [outfolder, outfolder, trilegal_version, l, b, area, filterset, magnum, maglim,
               binaries, mainparams, webserver, trilegal_version]
    cmd = ("wget -o {}/lixo -O {}/tmpfile --post-data='submit_form=Submit&trilegal_version={}"
           "&gal_coord=1&gc_l={}&gc_b={}&eq_alpha=0&eq_delta=0&field={}&photsys_file="
           "tab_mag_odfnew%2Ftab_mag_{}.dat&icm_lim={}&mag_lim={}&mag_res=0.1&"
           "binary_kind={}&{}' {}/cgi-bin/trilegal_{}").format(*cmdargs)
    complete = False
    while not complete:
        notconnected = True
        busy = True
        print("TRILEGAL is being called with \n l={} deg, b={} deg, area={} sqrdeg\n "
              "Av={} with {} fractional r.m.s. spread \n in the {} system, complete down to "
              "mag={} in its {}th filter, use_binaries set to {}.".format(*args))
        sp.Popen(cmd, shell=True).wait()
        if (os.path.exists('{}/tmpfile'.format(outfolder)) and
                os.path.getsize('{}/tmpfile'.format(outfolder)) > 0):
            notconnected = False
        else:
            print("No communication with {}, will retry in 2 min".format(webserver))
            time.sleep(120)
            return "nocomm"
        if not notconnected:
            with open('{}/tmpfile'.format(outfolder), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'The results will be available after about 2 minutes' in line:
                    busy = False
                    break
            sp.Popen('rm -f {}/lixo {}/tmpfile'.format(outfolder, outfolder), shell=True)
            if not busy:
                filenameidx = line.find('<a href=../tmp/') + 15
                fileendidx = line[filenameidx:].find('.dat')
                filename = line[filenameidx:filenameidx+fileendidx+4]
                print("retrieving data from {} ...".format(filename))
                while not complete:
                    time.sleep(40)
                    modcmd = 'wget -o {}/lixo -O {}/{} {}/tmp/{}'.format(
                        outfolder, outfolder, filename, webserver, filename)
                    sp.Popen(modcmd, shell=True).wait()
                    if os.path.getsize('{}/{}'.format(outfolder, filename)) > 0:
                        with open('{}/{}'.format(outfolder, filename), 'r') as f:
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
    sp.Popen('mv {}/{} {}'.format(outfolder, filename, outfile), shell=True).wait()
    print('results copied to {}'.format(outfile))

    return "good"


def get_AV_infinity(ra, dec, frame='icrs'):
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
    AV : numpy.ndarray
        Extinction at infinity as given by the SFD dust maps for the chosen sky
        coordinates.
    """
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    coords = SkyCoord(ra, dec, unit='deg', frame=frame).transform_to('galactic')

    sfd_ebv = dustmaps.sfd.SFDQuery()
    AV = 2.742 * sfd_ebv(coords)

    return AV
