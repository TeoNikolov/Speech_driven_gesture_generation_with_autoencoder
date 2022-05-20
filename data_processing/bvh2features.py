# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

import glob
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *

import joblib as jl
import glob


def extract_joint_angles(bvh_dir, files, dest_dir, pipeline_dir, fps):
    p = BVHParser()

    data_all = list()
    for f in files:
        ff = os.path.join(bvh_dir, f)
        print(ff)
        data_all.append(p.parse(ff))

    data_pipe = Pipeline([
       #('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
       #('mir', Mirror(axis='X', append=True)),
       ('exp', MocapParameterizer('expmap')),
       #('root', RootTransformer('hip_centric')),
       ('np', Numpyfier())
    ])


    out_data = data_pipe.fit_transform(data_all)
    
    # the datapipe will append the mirrored files to the end
    assert len(out_data) == len(files)
    
    jl.dump(data_pipe, os.path.join(pipeline_dir + 'data_pipe.sav'))
        
    fi=0
    for f in files:
        ff = os.path.join(dest_dir, f)
        print(ff)
        np.savez(ff[:-4] + ".npz", clips=out_data[fi])
        #np.savez(ff[:-4] + "_mirrored.npz", clips=out_data[len(files)+fi])
        fi=fi+1



if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--bvh_dir', '-orig', required=True,
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', required=True,
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
                        help="Path where the motion data processing pipeline will be stored")

    params = parser.parse_args()

    files = []
    # Go over all BVH files
    print("Going to pre-process the following motion files:")
    files = sorted([f for f in glob.iglob(params.bvh_dir+'/*.bvh')])

    extract_joint_angles(params.bvh_dir, files, params.dest_dir, params.pipeline_dir , fps=30)
