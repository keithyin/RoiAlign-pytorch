from functions import roi_align

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    x = torch.ones(2, 1, 6, 6).cuda()
    x[0, :, :3, :3] = 0

    x = Variable(x, requires_grad=True)

    rois = Variable(torch.FloatTensor([[0, 5, 5, 154, 154], [0, 1, 1, 100, 100], [1, 1, 1, 100, 100]]).cuda())

    res = roi_align.roi_align_op(x, rois, (5, 5), 1 / 32.)

    (torch.mean(res)).backward()

    # TODO: known bug , torch.sum(res).backward() the gradient will not compute correctly, haven't figure out why..
    print(x.grad)
