from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ckpt_dir', type=str,default = 'None')
        parser.add_argument('--vanilla', action='store_const', const=1, default=0)
        parser.add_argument('--is_QBATrain', action='store_const', const=0, default=1)
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        return parser