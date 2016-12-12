require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'

local function createModelD(input_size)
    model=nn.Sequential()
    model:add(nn.SpatialConvolution(1,64,3,3,1,1,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialConvolution(64,64,3,3,2,2,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.SpatialConvolution(128,128,3,3,2,2,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.SpatialConvolution(256,256,3,3,2,2,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.SpatialConvolution(256,512,3,3,1,1,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(nn.SpatialConvolution(512,512,3,3,2,2,1,1))
    model:add(nn.LeakyReLU(0.2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(nn.SpatialConvolution(512,1024,3,3,2,2,1,1))
    model:add(nn.LeakyReLU(0.2))
    kernel_size=input_size/32
    model:add(nn.SpatialConvolution(1024,1,kernel_size,kernel_size))
    model:add(nn.Sigmoid())
    model:add(nn.View(1))
    return model
end

return createModelD