require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'

local function createModel()
      local function bottleneck()
          local convs=nn.Sequential()
          convs:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
          convs:add(nn.SpatialBatchNormalization(64))
          convs:add(nn.ReLU(true))
          convs:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
          convs:add(nn.SpatialBatchNormalization(64))
          local shortcut=nn.Identity()
          return nn.Sequential():add(nn.ConcatTable():add(convs):add(shortcut)):add(nn.CAddTable(true))
      end

    local function layer(count)
      local s=nn.Sequential()
      for i=1,count do
        s:add(bottleneck())
      end
      return s
    end

    model=nn.Sequential()
    model:add(nn.SpatialConvolution(1,64,3,3,1,1,1,1))
    model:add(nn.ReLU())
    model:add(layer(15))
    model:add(nn.SpatialFullConvolution(64,64,3,3,2,2,1,1))
    model:add(nn.ReLU())
    model:add(nn.SpatialFullConvolution(64,64,3,3,2,2,1,1))
    model:add(nn.ReLU())
    model:add(nn.SpatialFullConvolution(64,1,3,3,1,1,1,1))
    return model
end

return createModel