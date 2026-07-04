"""Utility functions"""

import datetime
import os

def update_timestring():
    "Update identifying timestring"
    now = datetime.datetime.now()
    timestring = '%.4d_%.2d_%.2d_%.2d_%.2d_%.2d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    return timestring

def make_directories(p):
    "Make directories"
    if not os.path.exists(p.run_info_dir):
        os.makedirs(p.run_info_dir)
    if not os.path.exists(p.confusion_dir):
        os.makedirs(p.confusion_dir)
    if not os.path.exists(p.res_csv_deploy):
        os.makedirs(p.res_csv_deploy)
    if not os.path.exists(p.models_dir):
        os.makedirs(p.models_dir)
    if not os.path.exists(p.ckpt_dir):
        os.makedirs(p.ckpt_dir)
    if not os.path.exists(p.tb_log_dir):
        os.makedirs(p.tb_log_dir)

    if not os.path.exists(p.retrain_run_info_dir):
        os.makedirs(p.retrain_run_info_dir, exist_ok=False)
    if not os.path.exists(p.retrain_confusion_dir):
        os.makedirs(p.retrain_confusion_dir, exist_ok=False)
    # if not os.path.exists(p.retrain_res_csv_deploy):
    #     os.makedirs(p.retrain_res_csv_deploy, exist_ok=False)
    if not os.path.exists(p.retrain_models_dir):
        os.makedirs(p.retrain_models_dir, exist_ok=False)
    if not os.path.exists(p.retrain_ckpt_dir):
        os.makedirs(p.retrain_ckpt_dir, exist_ok=False)

def get_timepoint(s):
    t = s.split('/')[-1].split('_')[2]
    # print('timepoint', t)
    assert t[0] == 'T'
    assert t[1].isnumeric()
    return int(t[1:])