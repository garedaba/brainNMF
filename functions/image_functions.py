import os
import numpy as np
from nibabel.freesurfer import mghformat, io, MGHImage

def load_surf_data(left_surface_list, right_surface_list):
    """ Load individual surface data, and generate nxp matrix, where
        n = number of subjects and p = number of vertices in corresponding mask.

        ***NOTE: this assumes that the surfaces are in fsaverage5 space, in mgh format     ***
        ***   -if downsampled to fsaverage,fsaverage6,etc then adjust n_vert               ***
        ***   -if different format: CIFTI, GIFTI, then probably best to use something else!***

        Parameters
        ----------
        left_surface_list : List of surface files (.mgh) to load, including path to surface
        right_surface_list : Corresponding list of right hemipshere (rh) surface files to load, including path to surface

        Returns
        -------
        surf_data : numpy array (n_subjects, n_vertices)
        Vectorised surface data for all subjects"""


    n_vert = 10242  #number of vertices in fsaverage5 hemisphere

    surf_data = np.zeros((len(left_surface_list), n_vert*2))

    for it,filename in enumerate(left_surface_list):
        dat = mghformat.load(filename)
        surf_data[it,:n_vert] = np.squeeze(np.array(dat.get_fdata()))
        dat = mghformat.load(right_surface_list[it])
        surf_data[it,n_vert:] = np.squeeze(np.array(dat.get_fdata()))


    return surf_data

def save_surface_out(data, plot_path, hemi='lh', template='fsaverage5'):
        """save data to a freesurfer surface file.
        """
        surfname =os.environ['FREESURFER_HOME'] + '/subjects/' + template + '/surf/' + hemi + '.white.avg.area.mgh'
        surf_data = mghformat.load(surfname)
        surf_data.get_fdata()[:] = data.reshape(-1,1,1)

        comp=MGHImage((np.asarray(surf_data.get_fdata())),
                                        surf_data.affine ,
                                        extra=surf_data.extra,
                                        header=surf_data.header,
                                        file_map=surf_data.file_map)
        mghformat.save(comp, plot_path + '/' + hemi + '.out_data.mgh')
