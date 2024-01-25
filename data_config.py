
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = 'E:/CDDataset/LEVIR'
        elif data_name == 'DSIFN':
            self.label_transform = "norm"
            self.root_dir = 'E:/CDDataset/DSIFN_256'
        elif data_name == 'SYSU-CD':
            self.label_transform = "norm"
            self.root_dir = 'E:/CDDataset/SYSU-CD'
        elif data_name == 'LEVIR+':
            self.label_transform = "norm"
            self.root_dir = 'E:/CDDataset/LEVIR-CD+_256'
        elif data_name == 'BBCD':
            self.label_transform = "norm"
            self.root_dir = 'E:/CDDataset/Big_Building_ChangeDetection'
        elif data_name == 'GZ_CD':
            self.label_transform = "norm"
            self.root_dir = 'E:/CDDataset/GZ'
        elif data_name == 'WHU-CD':
            self.label_transform = "norm"
            self.root_dir = 'E:/CDDataset/WHU-CUT'
        elif data_name == 'test':
            self.label_transform = "norm"
            self.root_dir = 'E:/CDDataset/att_test_whu'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

