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
  ntrain = math.huge, 
  gpu=1,
  nThreads = 4,
  scale=4,
  loadSize=96,
  t_folder='',
  model_file='',
  result_path=''
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads,  opt)

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
    image.save(string.format('%s/raw_%04d.png',opt.result_path,cnt),image.toDisplayTensor(real[j]))
    image.save(string.format('%s/fake_%04d.png',opt.result_path,cnt),image.toDisplayTensor(fake[j]))
    cnt=cnt+1
  end
end




