import { memo, useState } from 'react'
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Shuffle, X } from 'lucide-react'
import { useGenerationStore } from '../../stores/generationStore'

const resolutionPresets = [
  { label: '1:1', width: 1024, height: 1024 },
  { label: '9:16', width: 576, height: 1024 },
  { label: '16:9', width: 1024, height: 576 },
  { label: '2:3', width: 768, height: 1152 },
  { label: '3:2', width: 1152, height: 768 },
  { label: '3:4', width: 768, height: 1024 },
]

const samplers = ['Flow', 'Flow_Unipc', 'Flow_DPM++']

function ParametersNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { params, setParams, nodeConnections } = useGenerationStore()
  const { deleteElements } = useReactFlow()
  const [isHovered, setIsHovered] = useState(false)
  
  // Check if Control node is connected to show Control Scale
  const isControlConnected = nodeConnections.isControlConnected

  const randomizeSeed = () => {
    setParams({ seed: Math.floor(Math.random() * 2147483647) })
  }

  return (
    <div 
      className={`min-w-[320px] rounded-lg border bg-card p-4 pr-16 relative ${selected ? 'border-primary' : 'border-border'}`}
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
      <div className="mb-3 text-sm font-semibold">{t('nodes.parameters')}</div>
      
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
