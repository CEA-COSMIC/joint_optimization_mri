import glob, os

from modopt.math.metrics import ssim, psnr
import numpy as np


def box_ssim(y_true, y_pred, mean_factor=0.7):
    return ssim(y_true, y_pred, y_true > mean_factor * np.mean(y_true))


def box_psnr(y_true, y_pred, mean_factor=0.7):
    return psnr(y_true, y_pred, y_true > mean_factor * np.mean(y_true))


def get_last_model(outdir):
    models = glob.glob(os.path.join(outdir, 'checkpoints', 'model*.h5'))
    optimizers = glob.glob(os.path.join(outdir, 'checkpoints', '*.pkl'))
    models_epochs = [int(os.path.basename(m).split('_')[1][1:]) for m in models]
    models = [models[i] for i in np.argsort(models_epochs)]
    models_epochs = np.sort(models_epochs)
    if '_E' in os.path.basename(optimizers[0]):
        # Backward compatibility
        opt_epochs = [int(os.path.basename(m).split('_')[1][1:-4]) for m in optimizers]
    else:
        opt_epochs = [models_epochs[-1]]
    optimizers = [optimizers[i] for i in np.argsort(opt_epochs)]
    opt_epochs = np.sort(opt_epochs)
    if models_epochs[-1] == opt_epochs[-1]:
        return os.path.basename(models[-1])
    else:
        return os.path.basename(models[np.nonzero(models_epochs == opt_epochs[-1])[0][0]])

def get_epoch_from_model_name(model_file, cfg, model=None):
    # MOST OF THIS CAN BE DEPRECATED
    def _get_type_from_total(total):
        if cfg['train']['type'] == 'segmented':
            if total < 2:
                type = 'trajectory' if total == 0 else 'recon_net'
            elif total < 2 + 2 * cfg['train']['segments'][0]:
                type = 'recon_net' if parts_done % 2 else 'trajectory'
            elif total < 2 + 2 * cfg['train']['segments'][0] +  2 * cfg['train']['segments'][1]:
                type = 'recon_net' if (total - 2 - 2 * cfg['train']['segments'][0]) % 2 else 'trajectory'
            else:
                type = 'joint'
        elif cfg['train']['type'] == 'multi_resolution':
            type = 'trajectory'
        else:
            type = cfg['train']['type'] # Covers for recon_net and new cases
        return type
    def _set_network_based_on_type(type):
        if type == 'trajectory':
            model.base_traj.trainable = True
            model.recon_net.trainable = False
        elif type == 'recon_net':
            model.base_traj.trainable = False
            model.recon_net.trainable = True
        else:
            model.base_traj.trainable = True
            model.recon_net.trainable = True
    current_epoch = int(os.path.basename(model_file).split('_')[1][1:]) - 1
    parts_done = current_epoch % cfg['train']['num_epochs']
    left_over = cfg['train']['num_epochs'] - parts_done
    total = current_epoch // cfg['train']['num_epochs']
    if model is not None:
        for i in range(total + 1):
            cur_type = _get_type_from_total(i)
            _set_network_based_on_type(cur_type)
    return current_epoch, left_over, model


METRIC_FUNCS = {
    'ssim': box_ssim,
    'psnr': box_psnr,
}