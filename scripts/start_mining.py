import os, csv
import numpy as np
from photutils import datasets
from datetime import datetime
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils.psf.groupstars import DAOGroup
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.nddata import Cutout2D


#aprior image and camera parameters
seconds_per_pixel = 0.154
spectral_band = "V"
imagtype = "light"
naxis1 =  "4296"
naxis2 = "4102"

#folders
fits_store = "../fits/"
csv_output = "../csv/"

def main():
    print("Seeng extractor")
    for root, _dirs, files in os.walk( fits_store ):
        fitsfiles = [f for f in files if f.endswith(".fits")]
        print("Find {} fits files in {} ".format(len(fitsfiles), fits_store)) 
        print("[IMAGTYPE,  FILTER] ")
        for fits_file_name in fitsfiles:
            fits_file_path = os.path.join(root, fits_file_name)
            if "2018" in fits_file_name:
                checks = []
                try:
                    with fits.open(fits_file_path) as hdul:
                        date = datetime.strptime(hdul[0].header["DATE_OBS"], "%Y-%m-%dT%H:%M:%S.%f")
                        checks.append( imagtype in hdul[0].header["IMAGETYP"] )
                        checks.append( spectral_band in hdul[0].header["FILTER"] )
                        #print(checks)
                        if all(checks):
                            fwhm = 18  # Aprior seeng in Pix
                            prop = get_seeing( hdul[1].data, fwhm, 10)
                            if prop:
                                objects_propertis = np.array(prop)
                                seeng_x = objects_propertis[:, 0]
                                seeng_y = objects_propertis[:, 1]
                                seeng = str(np.median((seeng_x + seeng_y) / 2.0))
                                print(fits_file_name, date, seeng)
                                with open(f"../csv/{fits_file_name[:-4]}.csv", "a") as file:
                                    writer = csv.writer(file)
                                    writer.writerow( prop )
                except:
                    pass


def get_seeing(raw_image, fwhm, threshold):
    mean, median, std = sigma_clipped_stats(raw_image, sigma=3.0, iters=1)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std, exclude_border=True)    
    sources = daofind(raw_image - median)
    sources = select_sources( sources )

    X = np.arange(-25, 25, 1.0)
    Y = np.arange(-25, 25, 1.0)
    X, Y = np.meshgrid(X, Y)
    source_prop = []

    for i in range(len(sources)):
        position = (sources['xcentroid'].data[i], sources['ycentroid'].data[i] )
        shape = (50, 50)
        Z = Cutout2D(raw_image, position, shape).data

        #Find fist approximation of fitting parameters
        dx = 0
        dy = 0
        for j in range(50):
            if Z[j, 25] >= (sources["peak"][i] / 2.0):
                dx += 1
            if Z[25, j] >= (sources["peak"][i] / 2.0):
                dy += 1

        #Interpolation
        g2 = models.Gaussian2D(sources["peak"][i], x_mean=0, y_mean=0, x_stddev = dx*gaussian_fwhm_to_sigma, y_stddev = dy*gaussian_fwhm_to_sigma)
        gf = fitting.LevMarLSQFitter()
        gaus = gf(g2, X, Y, Z - median )
        seeng_x = seconds_per_pixel * gaussian_sigma_to_fwhm * gaus.x_stddev.value
        seeng_y = seconds_per_pixel * gaussian_sigma_to_fwhm * gaus.y_stddev.value
        pos_x = sources['xcentroid'].data[i] + gaus.x_mean.value
        pos_y = sources['xcentroid'].data[i] + gaus.y_mean.value
        theta = gaus.theta.value
        source_prop.append([seeng_x, seeng_y, pos_x, pos_y, theta])
        #print(seeng_x, seeng_y)
    return source_prop

def select_sources( sources ):
    #many very strange constants ))
    max_peak_lim = 600_000
    min_dist_lim = 100
    crd_min_lim = 150
    crd_max_lim = 3850
    flux_lim_min = 25
    roundness1_lim_max = 0.1
    roundness2_lim_max = 0.1
    
    x = sources['xcentroid'].data # Convert to np.array
    y = sources['ycentroid'].data
    peak = sources['peak'].data
    roundness1 = sources["roundness1"].data
    roundness2 = sources["roundness2"].data
    flux = sources["flux"].data

    bad_items = []
    
    for i in range(len(sources)):
        distances = np.sqrt( np.power(x - x[i], 2) + np.power(y - y[i], 2) )
        distances = np.delete(distances, i)
        min_dist = distances.min()
        
        checks = []
        checks.append( x[i] > crd_min_lim )
        checks.append( x[i] < crd_max_lim )
        checks.append( not (x[i] > 1900  and  x[i] < 2350) )  # gap between two chips 
        checks.append( y[i] > crd_min_lim )
        checks.append( y[i] < crd_max_lim )
        checks.append( peak[i] < max_peak_lim )
        checks.append( flux[i] > flux_lim_min )
        checks.append( roundness1[i] < roundness1_lim_max )
        checks.append( roundness2[i] < roundness2_lim_max )
        checks.append( min_dist > min_dist_lim)

        print(checks)
        
        if all(checks):
            pass
        else:
            bad_items.append(i)
            
    sources.remove_rows(bad_items)
    return sources

if __name__ == "__main__":
    main()

