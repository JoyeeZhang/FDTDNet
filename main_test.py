from tqdm import tqdm
import time
import csv
from Network import *
from data_loader import *
from optparse import OptionParser

def get_args():
    parser = OptionParser()
    parser.add_option('-f', '--file', default='hello you need me', help='hello you need me')
    parser.add_option('-e', '--result', dest='result', default="test_details/radius32_results_epoch400_newlr1e-2_data1212_snr20_norm_LSTM1218/", help='folder of results')
    parser.add_option('-r', '--root', dest='root', default='D:/zhangjiaying/shear_modulus_estimation/', help='root directory')
    parser.add_option('-m', '--model', dest='model', default='model/model_weights_epoch400_newlr1e-2_data1212_snr20_norm_LSTM1218/weights.pth', help='folder for model/weights')
    parser.add_option('-i', '--input', dest='input', default="test_details/radius32/", help='folder of input')
    parser.add_option('-b', '--batch size', dest='batch_size', default=32, type='int', help='batch size')
    parser.add_option('-n', '--offsets', dest='offsets', default=8, type='int')
    parser.add_option('-o', '--field of view', dest='fov', default=0.2, type='float')
    options, args = parser.parse_args()
    return options

def run_test(dir_model, dir_input, dir_result, offsets, batch_size, fov):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net = Net().to(device)
    checkpoint = torch.load(dir_model, map_location=lambda storage, loc: storage.cuda(0))
    state_dict = checkpoint['state_dict']
    # 处理键的不匹配
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # 移除额外的 "module." 前缀
        new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict)
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)
    net.eval()

    loader = get_dataloader_for_test(dir_input, offsets, fov, batch_size)
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    net.eval()
    with torch.no_grad():
        time1 = 0
        num = 0
        for batch_idx, (input, gt, mfre, fov, index) in enumerate(loader):
            input, gt = input.to(device), gt.to(device)
            input = input.permute(4, 0, 1, 2, 3)
            output = net(input)
            input = input.squeeze(2).permute(1, 2, 3, 0)
            for th in range(0, input.shape[0]):
                time_start = time.time()
                input_numpy = input[th].squeeze(0).cpu().numpy()
                output_numpy = output[th].squeeze(0).cpu().numpy()
                gt_numpy = gt[th].squeeze(0).cpu().numpy()
                mfre_numpy = mfre[th].cpu().numpy()
                fov_numpy = fov[th].cpu().numpy()
                index_numpy = index[th].squeeze(0).squeeze(0).cpu().numpy()
                filename = dir_result + str(index_numpy).zfill(6) + '-results.mat'
                sio.savemat(filename, {'input': input_numpy, 'output': output_numpy, 'gt': gt_numpy,
                                       'mfre': mfre_numpy, 'fov': fov_numpy})
                print("Result saved as {}".format(filename))

                time_end = time.time()
                time1 = time1 + (time_end - time_start)
            num = num + 1
        print('totally cost', time1)

if __name__ == '__main__':
    args = get_args()
    run_test(dir_model=args.root + args.model,
             dir_input=args.root + args.input + "/",
             dir_result=args.root + args.result + "/",
             offsets=8,
             batch_size=32,
             fov=0.2)