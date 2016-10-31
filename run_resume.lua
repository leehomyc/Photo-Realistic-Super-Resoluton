require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'weight-init'
local G=require 'adversarial_G.lua'
local D=require 'adversarial_D.lua'
util = paths.dofile('util.lua')

opt = {
  dataset = 'folder',
  lr = 0.001,
  beta1 = 0.9,  
  batchSize=32,
  niter=250,
  loadSize=96,
  ntrain = math.huge, 
  name='super_resolution',
  gpu=1,
  nThreads = 4,
  scale=4,
  loadSize=96,
  t_folder='',
  model_folder='',
  D_path=''
  G_path=''
}
torch.manualSeed(1)

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

local real_label=1
local fake_label=0

local modelG=util.load(opt.G_path,opt.gpu)
local modelD=util.load(opt.D_path,opt.gpu)
local criterion = nn.BCECriterion() 
local criterion_mse = nn.MSECriterion()

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
   weightDecay=0.0001,
}
optimStateD = {
   learningRate = opt.lr*0.1,
   beta1 = opt.beta1,
    weightDecay=0.0001,
}

local input = torch.Tensor(opt.batchSize, 1, opt.loadSize/4, opt.loadSize/4) 
local real = torch.Tensor(opt.batchSize,1,opt.loadSize-3,opt.loadSize-3)
local real_uncropped = torch.Tensor(opt.batchSize,1,opt.loadSize,opt.loadSize)
local output = torch.Tensor(opt.batchSize,1,opt.loadSize-3,opt.loadSize-3)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local test = torch.Tensor(1, opt.loadSize-3, opt.loadSize-3) 
local test2 = torch.Tensor(1, opt.loadSize/4, opt.loadSize/4) 
local label = torch.Tensor(opt.batchSize)

if opt.gpu > 0 then
   require 'cunn'
   print('cunn used')
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  
   real=real:cuda(); 
   output=output:cuda();
   modelG=modelG:cuda()
   modelD=modelD:cuda()
   criterion:cuda()
   criterion_mse:cuda(); 
   label=label:cuda()
end

local parametersG, gradientsG = modelG:getParameters()
local parametersD,gradientsD= modelD:getParameters()

local fDx=function(x)
    if x ~= parametersD then
          parametersD:copy(x)
    end
    modelD:zeroGradParameters()

    real_uncropped,input= data:getBatch()
    real_uncropped=real_uncropped:cuda()
    real=real_uncropped[{{},{},{1,1+93-1},{1,1+93-1}}]
    real=real:cuda()

    label:fill(real_label)
    local output=modelD:forward(real)
    local errD_real=criterion:forward(output,label)
    local df_do = criterion:backward(output, label)
    modelD:backward(real,df_do)

    input=input:cuda()
    fake = modelG:forward(input)
    label:fill(fake_label)

    local output=modelD:forward(fake)
    local errD_fake=criterion:forward(output,label)
    local df_do = criterion:backward(output, label)
    modelD:backward(fake, df_do)

    errD = errD_real + errD_fake
    return errD, gradientsD
end

local fGx=function(x)
    modelG:zeroGradParameters()
    label:fill(real_label)
    local output=modelD.output
    
    input=input:cuda()
    real=real:cuda()

    errG = criterion:forward(output, label)
    errG_mse=criterion_mse:forward(fake,real)

    local df_do = criterion:backward(output, label)
    local df_do_mse=criterion_mse:backward(fake,real)

    local df_dg=modelD:updateGradInput(fake,df_do)
    modelG:backward(input,0.001*df_dg+0.999*df_do_mse)
    err_all=0.001*errG+0.999*errG_mse
    return err_all,gradientsG
end

local counter = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      optim.adam(fDx, parametersD, optimStateD)
      optim.adam(fGx, parametersG, optimStateG)

      counter = counter + 1
      print('count: '..counter)
      if counter % 10 == 0 then
          test:copy(real[1])
          print('test:')
          local real_rgb=test
          image.save(opt.name..counter..'_real.png',real_rgb)
          test2:copy(input[1])
          image.save(opt.name..counter..'_test.png',test2)
          fake[fake:gt(1)]=1
          fake[fake:lt(0)]=0
          test:copy(fake[1])
          image.save(opt.name..counter..'_fake.png',test)
      end
      
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f '
                   .. '  Err_G: %.4f Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, 
                 err_all and err_all or -1, errD and errD or -1))
      end
   end
   --paths.mkdir('/media/DATA/MODELS/SUPER_RES/checkpoints')
   parametersD, gradientsD= nil, nil
   parametersG, gradientsG = nil, nil
   util.save(opt.model_folder .. opt.name .. '_adversarial_G_' .. epoch, modelG, opt.gpu)
   util.save(opt.model_folder .. opt.name .. '_adversarial_D_' .. epoch, modelD, opt.gpu)
   parametersG, gradientsG = modelG:getParameters()
   parametersD, gradientsD=modelD:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end




