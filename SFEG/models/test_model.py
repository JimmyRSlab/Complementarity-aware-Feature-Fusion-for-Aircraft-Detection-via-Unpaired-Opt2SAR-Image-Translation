from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
        #                               opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = networks.define_G_SFEG(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                        opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        self.targets_mask_A = create_targets_mask(self.image_paths, self.real.shape, self.real.device)

    def forward(self):
        """Run forward pass."""
        # self.fake = self.netG(self.real)  # G(real)
        self.fake = self.netG(self.real, self.targets_mask_A)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass

import torch
from pathlib import Path
def create_targets_mask(image_paths, shape, device):
    """
    根据YOLO标注生成目标区域模板
    参数：
        image_paths: list[str], 图像路径列表（长度=batch_size）
        input_tensor: tensor, 输入图像张量 [b,c,h,w]
    返回：
        mask_tensor: tensor, 目标区域模板 [b,c,h,w]
    """
    b, c, h, w = shape
    mask_tensor = torch.zeros((b, 1, h, w), device=device)  # 创建初始掩码

    for batch_idx, img_path in enumerate(image_paths):
        # 1. 转换标注文件路径
        img_path = Path(img_path)

        # 处理不同操作系统的路径分隔符
        parts = list(img_path.parts)
        if "trainA" in parts:
            parts[parts.index("trainA")] = "labels_trainA"
        elif "testA" in parts:
            parts[parts.index("testA")] = "labels_testA"

        label_path = Path(*parts).with_suffix('.txt')

        # 2. 读取YOLO标注文件
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue  # 若无标注文件则保持全零

        # 3. 处理每个标注框
        for line in lines:
            line = line.strip().split()
            if len(line) < 5:
                continue  # 跳过无效行

            # 解析YOLO格式数据（类别, x_center, y_center, width, height）
            _, x_center, y_center, box_w, box_h = map(float, line[:5])

            # 转换为绝对坐标
            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            # 计算边界框坐标
            x0 = int(round(x_center - box_w / 2))
            y0 = int(round(y_center - box_h / 2))
            x1 = int(round(x_center + box_w / 2))
            y1 = int(round(y_center + box_h / 2))

            # 确保坐标在图像范围内
            x0 = max(0, min(x0, w - 1))
            y0 = max(0, min(y0, h - 1))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))

            # 4. 填充掩码
            mask_tensor[batch_idx, 0, y0:y1 + 1, x0:x1 + 1] = 1

    # 5. 通道维度匹配输入（假设c=1或需要复制到多个通道）
    # if c > 1:
    #     mask_tensor = mask_tensor.repeat(1, c, 1, 1)
    return mask_tensor