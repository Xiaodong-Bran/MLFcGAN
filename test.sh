set -ex
python test.py  \
--test_img_folder /home/nusuav/ICRA/methods/pix2pix-pytorch/dataset/flickr/test/source \
--netG unetsimple8Global \
--resume_netG_path ./pretrained/netG_best_model.pth \
--test_output_path ./infer_result/ \
--cuda

