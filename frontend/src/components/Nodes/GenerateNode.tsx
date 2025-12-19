import { memo, useState, useEffect, useCallback, useRef } from 'react'
import { Handle, Position, NodeProps, useReactFlow, useStore } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Play, Loader2, X } from 'lucide-react'
import { useGenerationStore } from '../../stores/generationStore'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

function GenerateNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { progress, startGeneration, lastGeneratedImage } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()
  const { deleteElements, getNode, getEdges, setNodes } = useReactFlow()
  const [isHovered, setIsHovered] = useState(false)
  
  // Track this node's active task ID
  const activeTaskIdRef = useRef<number | null>(null)
  
  // Subscribe to edges changes to track connections
  const edges = useStore((state) => state.edges)

  // Update node data when generation completes
  useEffect(() => {
    // Check if this node started the current generation and it's now complete
    if (
      activeTaskIdRef.current !== null &&
      progress.taskId === activeTaskIdRef.current &&
      !progress.isGenerating &&
      lastGeneratedImage
    ) {
      // Store generated image in node data for connected nodes (e.g., PreviewNode)
      setNodes((nodes) =>
        nodes.map((node) =>
          node.id === id
            ? { ...node, data: { ...node.data, generatedImage: lastGeneratedImage } }
            : node
        )
      )
      // Clear active task
      activeTaskIdRef.current = null
    }
  }, [progress.taskId, progress.isGenerating, lastGeneratedImage, id, setNodes])

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

  // Get mask image path from connected mask node
  const getMaskDataFromConnectedNode = useCallback(() => {
    const currentEdges = getEdges()
    const maskEdge = currentEdges.find(
      (e) => e.target === id && e.targetHandle === 'mask'
    )
    
    if (maskEdge) {
      const maskNode = getNode(maskEdge.source)
      return {
        maskImagePath: maskNode?.data?.maskImagePath || null,
        originalImagePath: maskNode?.data?.originalImagePath || null,
      }
    }
    return { maskImagePath: null, originalImagePath: null }
  }, [getEdges, getNode, id])

  // Get original image path from connected image node (for inpainting)
  const getImageDataFromConnectedNode = useCallback(() => {
    const currentEdges = getEdges()
    const imageEdge = currentEdges.find(
      (e) => e.target === id && e.targetHandle === 'image'
    )
    
    if (imageEdge) {
      const imageNode = getNode(imageEdge.source)
      return imageNode?.data?.imagePath || null
    }
    return null
  }, [getEdges, getNode, id])

  // Get prompt data from connected prompt node
  const getPromptDataFromConnectedNode = useCallback(() => {
    const currentEdges = getEdges()
    const promptEdge = currentEdges.find(
      (e) => e.target === id && e.targetHandle === 'prompt'
    )
    
    if (promptEdge) {
      const promptNode = getNode(promptEdge.source)
      if (promptNode?.data) {
        return {
          prompt: promptNode.data.prompt || '',
          promptKo: promptNode.data.promptKo || '',
        }
      }
    }
    return { prompt: '', promptKo: '' }
  }, [getEdges, getNode, id])

  // Get parameters data from connected parameters node
  const getParamsDataFromConnectedNode = useCallback(() => {
    const currentEdges = getEdges()
    const paramsEdge = currentEdges.find(
      (e) => e.target === id && e.targetHandle === 'params'
    )
    
    if (paramsEdge) {
      const paramsNode = getNode(paramsEdge.source)
      if (paramsNode?.data) {
        return {
          width: paramsNode.data.width ?? 1024,
          height: paramsNode.data.height ?? 1024,
          steps: paramsNode.data.steps ?? 8,
          seed: paramsNode.data.seed ?? null,
          sampler: paramsNode.data.sampler ?? 'Flow',
          controlScale: paramsNode.data.controlScale ?? 0.6,
        }
      }
    }
    // 기본값 반환
    return {
      width: 1024,
      height: 1024,
      steps: 8,
      seed: null,
      sampler: 'Flow',
      controlScale: 0.6,
    }
  }, [getEdges, getNode, id])

  // Check if control node is connected
  const isControlConnected = useCallback(() => {
    const currentEdges = getEdges()
    return currentEdges.some((e) => e.target === id && e.targetHandle === 'control')
  }, [getEdges, id])

  const handleGenerate = async () => {
    // Get prompt data from connected node
    const { prompt, promptKo } = getPromptDataFromConnectedNode()
    
    if (!prompt) {
      toastError('Prompt is required - connect a Prompt node')
      return
    }

    // Get parameters from connected node
    const params = getParamsDataFromConnectedNode()

    try {
      // Get control data from connected node
      const { controlImagePath, controlType } = getControlDataFromConnectedNode()
      // Get mask data from connected node (includes original image for inpainting)
      const { maskImagePath, originalImagePath: maskOriginalPath } = getMaskDataFromConnectedNode()
      // Get image data from connected image node
      const directImagePath = getImageDataFromConnectedNode()
      
      // Use original image from mask node or direct image connection
      const originalImagePath = maskOriginalPath || directImagePath
      
      const task = await generationApi.generate({
        prompt: prompt,
        prompt_ko: promptKo,
        width: params.width,
        height: params.height,
        num_inference_steps: params.steps,
        seed: params.seed,
        control_context_scale: isControlConnected() ? params.controlScale : undefined,
        sampler: params.sampler,
        control_type: controlType || undefined,
        control_image_path: controlImagePath,
        mask_image_path: maskImagePath,
        original_image_path: originalImagePath,
      })
      
      // Store task ID to track this node's generation
      activeTaskIdRef.current = task.id
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
        style={{ top: '12%' }}
        className="!bg-green-500 !w-3 !h-3"
      />
      <div className="absolute left-2 text-[9px] text-green-500 font-medium" style={{ top: '9%' }}>
        Prompt
      </div>
      <Handle
        type="target"
        position={Position.Left}
        id="params"
        style={{ top: '30%' }}
        className="!bg-purple-500 !w-3 !h-3"
      />
      <div className="absolute left-2 text-[9px] text-purple-500 font-medium" style={{ top: '27%' }}>
        Params
      </div>
      <Handle
        type="target"
        position={Position.Left}
        id="image"
        style={{ top: '48%' }}
        className="!bg-blue-500 !w-3 !h-3"
      />
      <div className="absolute left-2 text-[9px] text-blue-500 font-medium" style={{ top: '45%' }}>
        Image
      </div>
      <Handle
        type="target"
        position={Position.Left}
        id="control"
        style={{ top: '66%' }}
        className="!bg-orange-500 !w-3 !h-3"
      />
      <div className="absolute left-2 text-[9px] text-orange-500 font-medium" style={{ top: '63%' }}>
        Control
      </div>
      <Handle
        type="target"
        position={Position.Left}
        id="mask"
        style={{ top: '84%' }}
        className="!bg-pink-500 !w-3 !h-3"
      />
      <div className="absolute left-2 text-[9px] text-pink-500 font-medium" style={{ top: '81%' }}>
        Mask
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
