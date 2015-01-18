import os
import glob
import numpy as np
import re
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets import vstack

def load_openfmri_ds(root, subject, mask=None, filterfun=None, TR=3.0):
    
    ####Define helper functions for loading different parts of the data####
    def read_condition_keys(DSROOT):
        maxlength = 0
        def pick_fields(keys, line):
            fields = line.split()
            try:
                keys[fields[0]][fields[1]] = ' '.join(fields[2:])
            except KeyError:
                keys[fields[0]]={fields[1] : ' '.join(fields[2:])}
            return keys

        with open(os.path.join(DSROOT, 
                                    'models/model001/condition_key.txt'),'r') as keyfile:
            return reduce(pick_fields,keyfile,{})
        
    def parse_condition_onsets(path):
        condfiles=glob.glob(os.path.join(path, 'cond*'))
        timeline = []
        for cfile in condfiles:
            cond_name = os.path.basename(cfile).rstrip('.txt')
            with open(cfile,'r') as cfh:
                for line in cfh:
                    start, duration, weight = line.split()
                    timeline.append((float(start), float(duration), cond_name))
        timeline.sort()
        return timeline
        
    def extract_task_and_run(string):
        m=re.search('task([0-9]+)_run([0-9]+)', string)
        return int(m.group(1)), int(m.group(2))

    def load_run(runstring):
        ds=fmri_dataset(samples=os.path.join(root,subject,'BOLD',runstring,'bold.nii.gz'))
        task, run = extract_task_and_run(runstring)

        ds.sa['chunks'] = np.empty(len(ds))
        ds.sa.chunks.fill(run)
        ds.sa['task'] = np.empty(len(ds))
        ds.sa.task.fill(task)
        return ds
    
    def merge_conditions_onto_ds(ds, onsets):
        targets = np.chararray(ds.shape[0],itemsize=17)
        targets.fill('rest')
        for cond in onsets:
            start, duration, condition = cond
            startidx = int(start/TR)
            endidx = int((start+duration)/TR)
            targets[startidx:endidx+1] = condition_keys['task001'][condition]
        ds.sa['targets']=targets
    
    ##Actual data loading begins here
    condition_keys = read_condition_keys(root)
    
    allruns = map(lambda x: os.path.basename(x),
                glob.glob(os.path.join(root, subject,'BOLD/task*')))
    
    if filterfun:
        allruns = filter(filterfun,allruns)
    
    alldata=[]
    for run in allruns:
        ds=load_run(run)
        onsets=parse_condition_onsets(os.path.join(root,subject,'model/model001/onsets/',run))
        merge_conditions_onto_ds(ds,onsets)
        alldata.append(ds)
        
    merged = vstack(alldata)
    merged.a.update(alldata[0].a)
    return merged