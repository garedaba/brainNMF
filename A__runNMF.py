"""A__runNMF.py

Perform NMF decomposition of cortical surface area and thickness maps

Configuration options in config.yaml

"""
import glob, os

import numpy as np
import pandas as pd

from functions.image_functions import load_surf_data, save_surface_out
from NMF.nmf import ARDPNMF, normalise_features

from scipy.stats.mstats import winsorize
from neuroCombat import neuroCombat
import yaml


def main():

    # load parameters
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    print('')
    print('---------------------------------------------------------')
    print('Configuration:')
    print(yaml.dump(cfg, default_flow_style=False, default_style=''))
    print('---------------------------------------------------------')
    print('')

    # set paths
    datapath = cfg['paths']['datapath']
    metricpath = datapath + 'metrics/'
    outpath = cfg['paths']['outDir']

    # other params
    preprocessing_params = cfg['preprocessing']
    wins = 'Wins' if preprocessing_params['winsorise'] else 'NoWins'
    cbt = 'Combat' if preprocessing_params['combat'] else 'noCombat'
    unorm = 'Norm' if preprocessing_params['unit_norm'] else 'noNorm'

    nmf_params = cfg['nmf']
    subj = nmf_params['decomp_data']

    # load meta data
    meta_data = pd.read_csv(datapath + 'subject-data.csv')

    # load images
    for metric in ['thickness', 'area']:
        print('---------------------------------------------------------')
        print('processing: {:} data'.format(metric))
        print('')

        # load metric data
        lh_files = sorted(glob.glob(metricpath + '*.lh.' + metric + '*.mgh'))
        rh_files = sorted(glob.glob(metricpath + '*.rh.' + metric + '*.mgh'))

        surface_data = load_surf_data(lh_files, rh_files)

        # only include subjects with meta data (QC'd)
        subjects_with_images = [sub.split('/')[-1].split('.')[0] for sub in lh_files]
        include_subjects = np.isin(subjects_with_images, meta_data.participant_id)

        # remove vertices <=0
        surface_data[surface_data<0]=0
        zero_data = np.min(surface_data, axis=0)<=0
        surface_data = surface_data[:,zero_data==0]

        # remove excluded subjects
        surface_data = surface_data[include_subjects,:]

        # log if area
        if metric=='area':
            print('log-transform for area data')
            print('')
            surface_data = np.log1p(surface_data)

        if preprocessing_params['winsorise']:
            print('flattening top and bottom 1% of values')
            print('')
            surface_data = winsorize(surface_data, limits=(.01), axis=1).data

        if preprocessing_params['combat']:
            print('performing comBat harmonisation')
            print('')
            # combat
            surface_data = neuroCombat(surface_data, meta_data, 'scanner', discrete_cols=['diagnosis', 'male'], continuous_cols=['age'])
            # remove vertices <=0
            surface_data[surface_data<0]=0

        if preprocessing_params['unit_norm']:
            print('normalising individual data to unit norm')
            print('')
            surface_data  = normalise_features(surface_data)

        if subj == 'hc':
            print('performing NMF using healthy control data ONLY')
            surface_data = surface_data[meta_data.diagnosis=='CONTROL',:]

        if os.path.exists('{:}{:}-nmf-loadings-{:}-{:}-{:}-{:}.csv'.format(outpath, metric, wins, cbt, unorm, subj)):
            print('')
            print('*********************************************************')
            print('NMF for {:} already performed. See GeneratedData/'.format(metric))
            print('*********************************************************')

        else:
            print('')
            print('*********************************************************')
            print('RUNNING NMF')
            print('*********************************************************')
            nmf_results = ARDPNMF(surface_data.T, nmf_params['init_comps'],
                                                  init=nmf_params['init'],
                                                  maxIter=nmf_params['n_iter'],
                                                  tol=1e-6)

            print('')
            # components
            basis_components = nmf_results['W']
            basis_components = basis_components[:,np.argsort(-nmf_results['norms'])]
            basis_components = basis_components[:,nmf_results['norms']>0]

            # subject weights
            subject_loadings = basis_components.T.dot(surface_data.T).T

            # save out
            print('---------------------------------------------------------')
            print('saving NMF loadings to: {:}{:}-nmf-loadings-{:}-{:}-{:}-{:}.csv'.format(outpath, metric, wins, cbt, unorm, subj))
            column_labels = [('{:}{:02d}'.format(metric, i+1)) for i in np.arange(np.shape(subject_loadings)[1])]
            nmf_loadings = pd.concat((meta_data, pd.DataFrame(subject_loadings, columns=column_labels)), axis=1)
            nmf_loadings.to_csv('{:}{:}-nmf-loadings-{:}-{:}-{:}-{:}.csv'.format(outpath, metric, wins, cbt, unorm, subj), index=False)
            print('---------------------------------------------------------')
            print('saving NMF weights to: {:}{:}-nmf-components-{:}-{:}-{:}-{:}.npy'.format(outpath, metric, wins, cbt, unorm, subj))
            print('zero vector to: {:}{:}-nmf-zero-data.npy'.format(outpath, metric))
            np.save('{:}{:}-nmf-components-{:}-{:}-{:}-{:}.npy'.format(outpath, metric, wins, cbt, unorm, subj), basis_components)
            np.save('{:}{:}-nmf-zero-data.npy'.format(outpath, metric), zero_data)
            print('---------------------------------------------------------')
            print('')

            # save surface components
            print('saving NMF component maps')
            os.makedirs(outpath + 'surface_components', exist_ok=True)
            for comp in np.arange(np.shape(basis_components)[1]):
                comp_data = np.zeros(np.shape(zero_data))
                comp_data[~zero_data] = basis_components[:,comp]
                save_surface_out(comp_data[:10242], outpath + 'surface_components', hemi='lh', template='fsaverage5')
                os.rename(outpath + 'surface_components/lh.out_data.mgh', outpath + 'surface_components/lh.comp{:02d}-{:}-{:}-{:}-{:}-{:}.mgh'.format(comp+1, metric, wins, cbt, unorm, subj))
                save_surface_out(comp_data[10242:], outpath + 'surface_components', hemi='rh', template='fsaverage5')
                os.rename(outpath + 'surface_components/rh.out_data.mgh', outpath + 'surface_components/rh.comp{:02d}-{:}-{:}-{:}-{:}-{:}.mgh'.format(comp+1, metric, wins, cbt, unorm, subj))
            print('see ' + outpath + 'surface_components/?h.comp??.*.mgh')
            print('combine into 4D file using Freesurfer')

if __name__ == '__main__':
    main()
