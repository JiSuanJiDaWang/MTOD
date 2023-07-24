

class Option():
    def __init__(self, is_train=True):
        self.Cuda = True

        self.classes_path = 'model_data/cityscapes.txt'
        self.model_path = 'model_data/mobilenet_v2-b0353104.pth'
        self.input_shape = [512, 512]
        self.anchors_size = [30, 60, 111, 162, 213, 264, 315]
        self.train_annotation_path = "/content/drive/MyDrive/MTDD/train100.txt"
        self.val_annotation_path = "/content/drive/MyDrive/MTDD/val100.txt"

        self.epoch_step = 50
        # training information
        # the weight of each task: equal, dwa, uncert
        self.weight = 'uncert'

        self.batch_size = 4
        self.pretrained = True
        self.temp = 2.0
        # learning rate
        self.init_lr = 1e-3
        # save the parameters every peropd
        self.save_period = 10

        self.save_dir = 'hardfrom200'
        self.eval_flag = True
        self.eval_period = 10

        self.num_workers = 4
