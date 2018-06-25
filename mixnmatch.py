import numpy as np
import yaml
import pickle

studies = ['helmi', 'massari', 'kallivayalil']
names = []
study_from = []
MC_set = []

for study in studies:
    dwarf_file = 'data/dwarfs/'+study+'.yaml'
    with open(dwarf_file, 'r') as f:
        dwarfs = yaml.load(f)
    study_names = list(dwarfs.keys())
    MC_dwarfs = np.load('data/sampling/'+study+'_converted.npy')

    for name in study_names:
        if name not in names:
            names.append(name)
            study_from.append(study)
            MC_set.append(MC_dwarfs[study_names.index(name)])
MC_set = np.array(MC_set)

np.save('data/sampling/HMK', MC_set)

map = {'name': names, 'study': study_from}
with open('data/sampling/HMK_key.pkl', 'wb') as f:
    pickle.dump(map, f, protocol=pickle.HIGHEST_PROTOCOL)
