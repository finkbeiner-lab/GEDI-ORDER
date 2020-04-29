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
        os.mkdir(p.run_info_dir)
    if not os.path.exists(p.confusion_dir):
        os.mkdir(p.confusion_dir)
    if not os.path.exists(p.res_csv_deploy):
        os.mkdir(p.res_csv_deploy)
    if not os.path.exists(p.models_dir):
        os.mkdir(p.models_dir)
    if not os.path.exists(p.ckpt_dir):
        os.mkdir(p.ckpt_dir)

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