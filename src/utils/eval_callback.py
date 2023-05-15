import os
import numpy as np

import mindspore.ops as ops
from mindspore.train.callback import Callback
from mindspore import save_checkpoint


class EvalCallBack(Callback):
    """Precision verification using callback function."""

    def __init__(self, models, eval_dataset, eval_per_epochs, epochs_per_eval):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.epochs_per_eval = epochs_per_eval

    def on_train_epoch_end(self, run_context):
        """ evaluate during training """
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epochs == 0:
            val_loss = self.models.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epochs_per_eval["epoch"].append(cur_epoch)
            self.epochs_per_eval["val_loss"].append(val_loss)
            print(f"Epoch: {cur_epoch}, val_loss: {val_loss['val_loss']}")

    def get_dict(self):
        """ get eval dict"""
        return self.epochs_per_eval


class SegEvalCallback(Callback):
    """Callback for inference while training. Dataset cityscapes."""
    def __init__(self, loader, net, num_classes=19, ignore_label=255,
                 start_epoch=0, save_path=None, interval=1, rank=0):
        super(SegEvalCallback, self).__init__()
        self.loader = loader
        self.net = net
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.start_epoch = start_epoch
        self.save_path = save_path
        self.interval = interval
        self.best_miou = 0
        self.device_id = rank
        self.save_path = os.path.join(self.save_path, f"ckpt_{self.device_id}")
        if not os.path.exists(os.path.realpath(self.save_path)):
            os.makedirs(os.path.realpath(self.save_path))

    def on_train_epoch_end(self, run_context):
        """Epoch end."""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch >= self.start_epoch:
            if (cur_epoch - self.start_epoch) % self.interval == 0:
                self.net.set_train(False)

                miou = self.inference()
                if miou > self.best_miou:
                    self.best_miou = miou
                    if self.save_path:
                        file_path = os.path.join(self.save_path, "best_model.ckpt")
                        save_checkpoint(self.net, file_path)
                print("=== epoch: {:4d}, device id: {:2d}, best miou: {:6.4f}, miou: {:6.4f}".format(
                    cur_epoch, self.device_id, self.best_miou, miou), flush=True)
                self.net.set_train(True)

    def inference(self):
        """Cityscapes inference."""
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for batch in self.loader:
            image, label = batch
            shape = label.shape
            pred = self.net(image)
            pred = ops.Exp()(pred)

            confusion_matrix += self.get_confusion_matrix(label, pred, shape,
                                                          self.num_classes, self.ignore_label)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        iou_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_iou = iou_array.mean()
        return mean_iou

    def get_confusion_matrix(self, label, pred, shape, num_class, ignore=255):
        """
        Calcute the confusion matrix by given label and pred.
        """
        output = pred.asnumpy().transpose(0, 2, 3, 1)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        seg_gt = np.asarray(label.asnumpy()[:, :shape[-2], :shape[-1]], dtype=np.int32)

        ignore_index = seg_gt != ignore
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]

        index = (seg_gt * num_class + seg_pred).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((num_class, num_class))

        for i_label in range(num_class):
            for i_pred in range(num_class):
                cur_index = i_label * num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred] = label_count[cur_index]
        return confusion_matrix
