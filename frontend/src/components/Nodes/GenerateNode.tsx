import { memo, useState, useEffect, useCallback } from 'react'
import { Handle, Position, NodeProps, useReactFlow, useStore } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Play, Loader2, X } from 'lucide-react'
import { useGenerationStore } from '../../stores/generationStore'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

function GenerateNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { params, progress, startGeneration, updateConnectionsFromEdges, nodeConnections } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()
  const { deleteElements, getNode, getEdges } = useReactFlow()
  const [isHovered, setIsHovered] = useState(false)
  
  // Subscribe to edges changes to track connections
  const edges = useStore((state) => state.edges)
  
  // Update connection state when edges change
  useEffect(() => {
    updateConnectionsFromEdges(edges, id)
  }, [edges, id, updateConnectionsFromEdges])

  // Get control image path from connected control node
  const getControlDataFromConnectedNode = useCallback(() => {
    const currentEdges = getEdges()
    const controlEdge = currentEdges.find(
      (e) => e.target === id && e.targetHandle === 'control'
    )
    
    if (controlEdge) {
      const controlNode = getNode(controlEdge.source)
      if (controlNode?.data) {
        return {
          controlImagePath: controlNode.data.controlImagePath || null,
          controlType: controlNode.data.controlType || null,
        }
      }
    }
    return { controlImagePath: null, controlType: null }
  }, [getEdges, getNode, id])

  const handleGenerate = async () => {
    if (!params.prompt) {
      toastError('Prompt is required')
      return
    }

    try {
      // Get control data from connected node
      const { controlImagePath, controlType } = getControlDataFromConnectedNode()
      
      const task = await generationApi.generate({
        prompt: params.prompt,
        prompt_ko: params.promptKo,
        width: params.width,
        height: params.height,
        num_inference_steps: params.steps,
        seed: params.seed,
        control_context_scale: nodeConnections.isControlConnected ? params.controlScale : undefined,
        sampler: params.sampler,
        control_type: controlType || params.controlType,
        control_image_path: controlImagePath,
      })
      
      startGeneration(task.id)
      toastSuccess(t('common.success'), `Task ${task.id} started`)
    } catch (err) {
      toastError(t('errors.generationFailed'), String(err))
    }
  }

  const isGenerating = progress.isGenerating

  return (
    <div 
      className={`min-w-[220px] rounded-lg border bg-card p-4 pl-14 pr-14 relative ${selected ? 'border-primary' : 'border-border'}`}
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
      {/* Input handles */}
      <Handle
        type="target"
        position={Position.Left}
        id="prompt"
        style={{ top: '30%' }}
        className="!bg-green-500 !w-3 !h-3"
      />
      <div className="absolute left-2 text-[9px] text-green-500 font-medium" style={{ top: '27%' }}>
        Prompt
      </div>
      <Handle
        type="target"
        position={Position.Left}
        id="params"
        style={{ top: '50%' }}
        className="!bg-purple-500 !w-3 !h-3"
      />
      <div className="absolute left-2 text-[9px] text-purple-500 font-medium" style={{ top: '47%' }}>
        Params
      </div>
      <Handle
        type="target"
        position={Position.Left}
        id="control"
        style={{ top: '70%' }}
        className="!bg-orange-500 !w-3 !h-3"
      />
      <div className="absolute left-2 text-[9px] text-orange-500 font-medium" style={{ top: '67%' }}>
        Control
      </div>
      
      <div className="mb-3 text-sm font-semibold">{t('nodes.generate')}</div>
      
      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={isGenerating}
        className={`nodrag w-full flex items-center justify-center gap-2 rounded-md px-4 py-3 text-sm font-medium
          transition-all duration-200
          ${isGenerating 
            ? 'bg-primary/50 cursor-not-allowed' 
            : 'bg-primary text-primary-foreground hover:bg-primary/90 hover:scale-105'
          }`}
      >
        {isGenerating ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            <span>{progress.progress}%</span>
          </>
        ) : (
          <>
            <Play className="h-5 w-5" />
            {t('generate.generate')}
          </>
        )}
      </button>
      
      {/* Progress indicator */}
      {isGenerating && (
        <div className="mt-3">
          <div className="h-1 w-full rounded-full bg-muted overflow-hidden">
            <div 
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${progress.progress}%` }}
            />
          </div>
          <div className="mt-1 text-center text-xs text-muted-foreground">
            Step {progress.currentStep}/{progress.totalSteps}
          </div>
        </div>
      )}
      
      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="!bg-blue-500 !w-3 !h-3"
      />
      <div className="absolute right-2 top-[50%] -translate-y-1/2 text-[9px] text-blue-500 font-medium">
        Image
      </div>
    </div>
  )
}

export const GenerateNode = memo(GenerateNodeComponent)
