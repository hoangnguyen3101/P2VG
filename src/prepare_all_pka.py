import os
import sys
import subprocess
from glob import glob
from multiprocessing import Pool

INPUT_DIR = "/storage/hoangnv/dataset_pka_ttd"
OUTPUT_DIR = "/storage/hoangnv/dataset_pka_ttd_size"

def process_subject(subject_id):
    try:
        cmd = [
            "/home/hoangnv/miniconda3/envs/p2vg/bin/python", "/home/hoangnv/AICD_HA/SPINE_BASE/P2VG/src/prepare_pka_sample.py",
            "--subject_id", subject_id,
            "--input_dir", INPUT_DIR,
            "--output_dir", OUTPUT_DIR,
            "--mode", "preserve_axis",
            "--xy_size", "256",
            "--depth_size", "32"
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            print("Error processing {0}. Return code {1}. STDERR: {2}".format(subject_id, p.returncode, err.decode('utf-8', errors='ignore').strip()))
            return subject_id, False
            
        print("Successfully processed {0}".format(subject_id))
        return subject_id, True
    except Exception as e:
        print("Exception processing {0}: {1}".format(subject_id, e))
        return subject_id, False

if __name__ == "__main__":
    files = glob(os.path.join(INPUT_DIR, "sub-*_fused.nii.gz"))
    subjects = sorted(list(set([os.path.basename(f).split("_")[0] for f in files])))
    print("Found {0} subjects. Starting processing in parallel...".format(len(subjects)))
    
    pool = Pool(8)
    results = pool.map(process_subject, subjects)
    pool.close()
    pool.join()
    
    success = sum(1 for r in results if r[1])
    print("Done! Successfully processed {0}/{1} subjects.".format(success, len(subjects)))
    print("Output saved to: {0}".format(OUTPUT_DIR))
