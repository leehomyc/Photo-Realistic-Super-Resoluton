require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'cudnn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  dataset = 'folder', 
  batchSize=32,
  niter=250,
  fineSize=96,
  ntrain = math.huge, 
  gpu=1,
  nThreads = 4,
  scale=4,
  loadSize=96,
  test_folder='/media/harryyang/New Volume/vision-harry/mp4_videos/test_folder',
  model_file='/media/DATA/MODELS/SUPER_RES/checkpoints/adv_color_very_deep_good_init_D_smallerrate_adversarial_G_3',
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads, 4, opt)

modelG=util.load(opt.model_file,opt.gpu)

cnt=1
for i = 1, opt.niter do
  real_uncropped,input= data:getBatch()
  real=real_uncropped[{{},{},{1,1+93-1},{1,1+93-1}}]
  print(i)
  fake = modelG:forward(input)
  fake[fake:gt(1)]=1
  fake[fake:lt(0)]=0
  for j=1,opt.batchSize do
    image.save(string.format('/media/harryyang/New Volume/vision-harry/mp4_videos/test_folder_hao_res/video_raw/raw_%04d.png',cnt),image.toDisplayTensor(real[j]))
    image.save(string.format('/media/harryyang/New Volume/vision-harry/mp4_videos/test_folder_hao_res/bicubic/fake_%04d.png',cnt),image.toDisplayTensor(fake[j]))
    cnt=cnt+1
  end
end




