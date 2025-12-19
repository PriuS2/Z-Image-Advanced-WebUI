import { memo, useState, useCallback } from 'react'
import { Handle, Position, NodeProps, useReactFlow, useStore } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Shuffle, X, Link, ImageIcon } from 'lucide-react'

const resolutionPresets = [
  { label: '1:1', width: 1024, height: 1024 },
  { label: '9:16', width: 576, height: 1024 },
  { label: '16:9', width: 1024, height: 576 },
  { label: '2:3', width: 768, height: 1152 },
  { label: '3:2', width: 1152, height: 768 },
  { label: '3:4', width: 768, height: 1024 },
]

const samplers = ['Flow', 'Flow_Unipc', 'Flow_DPM++']

// 노드 데이터 타입 정의
interface ParametersNodeData {
  width?: number
  height?: number
  steps?: number
  seed?: number | null
  sampler?: string
  controlScale?: number
}

// 기본값
const defaultParams: Required<Omit<ParametersNodeData, 'seed'>> & { seed: number | null } = {
  width: 1024,
  height: 1024,
  steps: 8,
  seed: null,
  sampler: 'Flow',
  controlScale: 0.6,
}

function ParametersNodeComponent({ id, selected, data }: NodeProps<ParametersNodeData>) {
  const { t } = useTranslation()
  const { deleteElements, setNodes } = useReactFlow()
  const [isHovered, setIsHovered] = useState(false)
  
  // 노드 자체의 데이터 (글로벌 스토어 대신)
  const params = {
    width: data?.width ?? defaultParams.width,
    height: data?.height ?? defaultParams.height,
    steps: data?.steps ?? defaultParams.steps,
    seed: data?.seed ?? defaultParams.seed,
    sampler: data?.sampler ?? defaultParams.sampler,
    controlScale: data?.controlScale ?? defaultParams.controlScale,
  }

  // 노드 데이터 업데이트 함수
  const setParams = useCallback((newData: Partial<ParametersNodeData>) => {
    setNodes((nodes) =>
      nodes.map((node) =>
        node.id === id
          ? { ...node, data: { ...node.data, ...newData } }
          : node
      )
    )
  }, [id, setNodes])
  
  // Subscribe to edges changes to detect connections
  const edges = useStore((state) => state.edges)
  
  // Track the connected source node ID for image input
  const connectedImageSourceId = edges.find(
    (e) => e.target === id && e.targetHandle === 'image-input'
  )?.source || null
  
  // Track the connected source node ID for control input
  const connectedControlSourceId = edges.find(
    (e) => e.target === id && e.targetHandle === 'control-input'
  )?.source || null
  
  // Subscribe to the connected image node's dimensions
  const connectedImageData = useStore((state) => {
    if (!connectedImageSourceId) return null
    const sourceNode = state.nodeInternals.get(connectedImageSourceId)
    if (!sourceNode?.data) return null
    return {
      width: sourceNode.data.imageWidth as number | null,
      height: sourceNode.data.imageHeight as number | null,
    }
  })
  
  // Subscribe to the connected control node's dimensions
  const connectedControlData = useStore((state) => {
    if (!connectedControlSourceId) return null
    const sourceNode = state.nodeInternals.get(connectedControlSourceId)
    if (!sourceNode?.data) return null
    return {
      width: sourceNode.data.imageWidth as number | null,
      height: sourceNode.data.imageHeight as number | null,
    }
  })
  
  const isImageConnected = !!connectedImageSourceId
  const isControlConnected = !!connectedControlSourceId
  
  // Either image or control can be connected, but not both for resolution
  const activeConnectionData = connectedImageData || connectedControlData
  const hasValidDimensions = activeConnectionData?.width && activeConnectionData?.height
  const hasAnyConnection = isImageConnected || isControlConnected
  
  const applyImageResolution = () => {
    if (activeConnectionData?.width && activeConnectionData?.height) {
      // Round to nearest 64 (common requirement for image generation)
      const width = Math.round(activeConnectionData.width / 64) * 64
      const height = Math.round(activeConnectionData.height / 64) * 64
      setParams({ width: Math.max(256, Math.min(2048, width)), height: Math.max(256, Math.min(2048, height)) })
    }
  }

  const randomizeSeed = () => {
    setParams({ seed: Math.floor(Math.random() * 2147483647) })
  }

  return (
    <div 
      className={`min-w-[320px] rounded-lg border bg-card p-4 pl-12 pr-16 relative ${selected ? 'border-primary' : 'border-border'}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Delete button */}
      {isHovered && (
        <button
          onClick={() => deleteElements({ nodes: [{ id }] })}
          className="absolute -right-2 -top-2 z-10 rounded-full bg-destructive p-1 text-destructive-foreground 
            hover:bg-destructive/80 transition-colors shadow-md"
        >
          <X className="h-3 w-3" />
        </button>
      )}
      
      {/* Image Input handle - disabled if control is connected */}
      <Handle
        type="target"
        position={Position.Left}
        id="image-input"
        style={{ top: '30%' }}
        className={`!w-3 !h-3 ${
          isImageConnected 
            ? '!bg-green-500' 
            : isControlConnected 
              ? '!bg-gray-400 !cursor-not-allowed' 
              : '!bg-blue-500'
        }`}
        isConnectable={!isControlConnected}
      />
      <div 
        className={`absolute left-2 text-[9px] font-medium flex items-center gap-1 ${
          isImageConnected 
            ? 'text-green-500' 
            : isControlConnected 
              ? 'text-gray-400' 
              : 'text-blue-500'
        }`}
        style={{ top: '30%', transform: 'translateY(-50%)' }}
      >
        {isImageConnected && <Link className="h-2.5 w-2.5" />}
        Image
      </div>
      
      {/* Control Input handle - disabled if image is connected */}
      <Handle
        type="target"
        position={Position.Left}
        id="control-input"
        style={{ top: '70%' }}
        className={`!w-3 !h-3 ${
          isControlConnected 
            ? '!bg-green-500' 
            : isImageConnected 
              ? '!bg-gray-400 !cursor-not-allowed' 
              : '!bg-orange-500'
        }`}
        isConnectable={!isImageConnected}
      />
      <div 
        className={`absolute left-2 text-[9px] font-medium flex items-center gap-1 ${
          isControlConnected 
            ? 'text-green-500' 
            : isImageConnected 
              ? 'text-gray-400' 
              : 'text-orange-500'
        }`}
        style={{ top: '70%', transform: 'translateY(-50%)' }}
      >
        {isControlConnected && <Link className="h-2.5 w-2.5" />}
        Control
      </div>
      
      <div className="mb-3 text-sm font-semibold">{t('nodes.parameters')}</div>
      
      {/* Connected image/control resolution info */}
      {hasAnyConnection && hasValidDimensions && (
        <div className="mb-3 p-2 rounded-md bg-green-500/10 border border-green-500/30">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
              <ImageIcon className="h-3.5 w-3.5" />
              <span>
                {isImageConnected ? 'Image' : 'Control'}: {activeConnectionData?.width} × {activeConnectionData?.height}
              </span>
            </div>
            <button
              onClick={applyImageResolution}
              className="nodrag px-2 py-1 rounded text-xs bg-green-500 text-white hover:bg-green-600 transition-colors"
            >
              해상도 적용
            </button>
          </div>
        </div>
      )}
      
      {/* Resolution presets */}
      <div className="mb-3">
        <label className="mb-1 block text-xs text-muted-foreground">Resolution</label>
        <div className="flex flex-wrap gap-1">
          {resolutionPresets.map((preset) => (
            <button
              key={preset.label}
              onClick={() => setParams({ width: preset.width, height: preset.height })}
              className={`nodrag px-2 py-1 rounded text-xs transition-colors
                ${params.width === preset.width && params.height === preset.height
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>
      
      {/* Width/Height */}
      <div className="mb-3 grid grid-cols-2 gap-2">
        <div>
          <label className="mb-1 block text-xs text-muted-foreground">
            {t('generate.width')}
          </label>
          <input
            type="number"
            value={params.width}
            onChange={(e) => setParams({ width: parseInt(e.target.value) || 1024 })}
            className="nodrag w-full rounded-md border border-input bg-background px-2 py-1 text-sm"
            step={64}
            min={256}
            max={2048}
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-muted-foreground">
            {t('generate.height')}
          </label>
          <input
            type="number"
            value={params.height}
            onChange={(e) => setParams({ height: parseInt(e.target.value) || 1024 })}
            className="nodrag w-full rounded-md border border-input bg-background px-2 py-1 text-sm"
            step={64}
            min={256}
            max={2048}
          />
        </div>
      </div>
      
      {/* Steps */}
      <div className="mb-3">
        <label className="mb-1 block text-xs text-muted-foreground">
          {t('generate.steps')}: {params.steps}
        </label>
        <input
          type="range"
          value={params.steps}
          onChange={(e) => setParams({ steps: parseInt(e.target.value) })}
          className="nodrag w-full accent-primary"
          min={1}
          max={50}
        />
      </div>
      
      {/* Control Scale - Only shown when Control node is connected */}
      {isControlConnected && (
        <div className="mb-3">
          <label className="mb-1 block text-xs text-muted-foreground">
            {t('generate.controlScale')}: {params.controlScale.toFixed(2)}
          </label>
          <input
            type="range"
            value={params.controlScale * 100}
            onChange={(e) => setParams({ controlScale: parseInt(e.target.value) / 100 })}
            className="nodrag w-full accent-primary"
            min={0}
            max={100}
          />
        </div>
      )}
      
      {/* Seed */}
      <div className="mb-3">
        <label className="mb-1 block text-xs text-muted-foreground">
          {t('generate.seed')}
        </label>
        <div className="flex gap-2">
          <input
            type="number"
            value={params.seed ?? ''}
            onChange={(e) => setParams({ seed: e.target.value ? parseInt(e.target.value) : null })}
            placeholder={t('generate.randomSeed')}
            className="nodrag flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm"
          />
          <button
            onClick={randomizeSeed}
            className="nodrag rounded-md bg-secondary p-2 hover:bg-secondary/80 transition-colors"
            title={t('generate.randomSeed')}
          >
            <Shuffle className="h-4 w-4" />
          </button>
        </div>
      </div>
      
      {/* Sampler */}
      <div>
        <label className="mb-1 block text-xs text-muted-foreground">
          {t('generate.sampler')}
        </label>
        <select
          value={params.sampler}
          onChange={(e) => setParams({ sampler: e.target.value })}
          className="nodrag w-full rounded-md border border-input bg-background px-2 py-1 text-sm"
        >
          {samplers.map((sampler) => (
            <option key={sampler} value={sampler}>
              {sampler}
            </option>
          ))}
        </select>
      </div>
      
      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="!bg-purple-500 !w-3 !h-3"
      />
      <div className="absolute right-2 top-[50%] -translate-y-1/2 text-[9px] text-purple-500 font-medium">
        Params
      </div>
    </div>
  )
}

export const ParametersNode = memo(ParametersNodeComponent)
