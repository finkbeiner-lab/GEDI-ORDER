import datetime
import os

def update_timestring():
    now = datetime.datetime.now()
    timestring = '%.4d_%.2d_%.2d_%.2d_%.2d_%.2d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    return timestring

def make_directories(p):
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